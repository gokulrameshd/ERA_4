import sys
import os
import math
import numpy as np
import random
import time
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C") # Green
C_SENSOR_OFF= QColor("#BF616A") # Red

# Physics Tweaks
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 16
SENSOR_ANGLE = 45
SPEED = 1       
TURN_SPEED = 1
SHARP_TURN = 30  # Sharp turn angle for tight corners

# RL - TD3 Parameters
BATCH_SIZE = 512 # Try 256 for faster updates, or 512 for stability
GAMMA = 0.98
LR_ACTOR = 0.0005 # Slightly lower for stability (was 0.001)
LR_CRITIC = 0.001 # Keep higher for faster Q-learning
TAU = 0.01  # Increase for faster target network updates (was 0.005) # Polyak averaging coefficient for soft target updates (TD3 uses 0.005)
POLICY_NOISE = 0.2  # Keep standard # Noise added to target policy during critic update
NOISE_CLIP = 0.5  # Keep standard # Range to clip target policy noise
POLICY_FREQ = 2  # Keep at 2 (standard TD3) # Frequency of delayed policy updates (update actor every N steps)
EXPL_NOISE = 0.2    # Start higher, decay over time (was 0.1) # Exploration noise standard deviation
MAX_STEERING = 20.0  # Maximum steering angle in degrees (reduced to prevent tight circles)
MAX_THROTTLE = 2.0  # Maximum throttle/speed multiplier (e.g., 3x base speed)
MIN_THROTTLE = 0.5  # Minimum throttle (increased to ensure visible movement)
MAX_ACTION = np.array([MAX_STEERING, MAX_THROTTLE])  # Action bounds [steering, throttle]
MAX_CONSECUTIVE_CRASHES = 2  # Stop training after this many consecutive crashes

# Target Colors (for multiple targets)
TARGET_COLORS = [
    QColor(0, 255, 255),      # Cyan
    QColor(255, 100, 255),    # Magenta
    QColor(0, 255, 100),      # Green
    QColor(255, 150, 0),      # Orange
    QColor(100, 150, 255),    # Blue
    QColor(255, 50, 150),     # Pink
    QColor(150, 255, 50),     # Lime
    QColor(255, 255, 0),      # Yellow
]

# ==========================================
# 2. NEURAL NETWORK - TD3 Architecture
# ==========================================
class Actor(nn.Module):
    """TD3 Actor Network: outputs [steering, throttle]"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # Changed from BatchNorm1d to LayerNorm
        self.layer_2 = nn.Linear(256, 512)
        self.ln2 = nn.LayerNorm(512)
        self.layer_3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)
        self.layer_4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)
        self.layer_5 = nn.Linear(256, action_dim)
        # Store action bounds separately for steering (symmetric) and throttle (asymmetric)
        self.max_steering = max_action[0]
        self.min_throttle = MIN_THROTTLE
        self.max_throttle = max_action[1]
        
        # Initialize last layer to bias throttle towards higher values initially
        # This helps the car start moving from the beginning
        with torch.no_grad():
            # Bias the throttle output (second dimension) towards positive values
            # tanh(0.5) ≈ 0.46, which maps to ~0.73 throttle (good starting point)
            self.layer_5.bias[1].fill_(0.5)  # Bias throttle output towards positive
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.ln1(x)  # LayerNorm before activation
        x = torch.relu(x)
        
        x = self.layer_2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        
        x = self.layer_3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        
        x = self.layer_4(x)
        x = self.ln4(x)
        x = torch.relu(x)
        
        x = torch.tanh(self.layer_5(x))  # Output in [-1, 1]

        # Scale steering: [-1, 1] -> [-MAX_STEERING, MAX_STEERING]
        steering = x[:, 0:1] * self.max_steering
        
        # Scale throttle: [-1, 1] -> [MIN_THROTTLE, MAX_THROTTLE]
        throttle = (x[:, 1:2] + 1.0) / 2.0  # [0, 1]
        throttle = throttle * (self.max_throttle - self.min_throttle) + self.min_throttle
        
        return torch.cat([steering, throttle], dim=1)

class Critic(nn.Module):
    """TD3 Twin Critic Network: two Q-networks to reduce overestimation"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Critic Network
        self.layer_1 = nn.Linear(state_dim + action_dim, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 256)
        self.layer_4 = nn.Linear(256, 128)
        self.layer_5 = nn.Linear(128, 1)
        
        # Second Critic Network
        self.layer_6 = nn.Linear(state_dim + action_dim, 128)
        self.layer_7 = nn.Linear(128, 256)
        self.layer_8 = nn.Linear(256, 256)
        self.layer_9 = nn.Linear(256, 128)
        self.layer_10 = nn.Linear(128, 1)
        
    def forward(self, x, u):
        """Returns both Q-values"""
        xu = torch.cat([x, u], 1)
        # Q1
        q1 = torch.relu(self.layer_1(xu))
        q1 = torch.relu(self.layer_2(q1))
        q1 = torch.relu(self.layer_3(q1))
        q1 = torch.relu(self.layer_4(q1))
        q1 = self.layer_5(q1)
        # Q2
        q2 = torch.relu(self.layer_6(xu))
        q2 = torch.relu(self.layer_7(q2))
        q2 = torch.relu(self.layer_8(q2))
        q2 = torch.relu(self.layer_9(q2))
        q2 = self.layer_10(q2)
        return q1, q2
    
    def Q1(self, x, u):
        """Returns only Q1 (used for actor update)"""
        xu = torch.cat([x, u], 1)
        q1 = torch.relu(self.layer_1(xu))
        q1 = torch.relu(self.layer_2(q1))
        q1 = torch.relu(self.layer_3(q1))
        q1 = torch.relu(self.layer_4(q1))
        q1 = self.layer_5(q1)
        return q1

# ==========================================
# 3. PHYSICS & LOGIC
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # RL Init - TD3
        self.input_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        self.action_dim = 2  # [steering, throttle]
        self.max_action = MAX_ACTION  # Now an array [MAX_STEERING, MAX_THROTTLE]
        
        # TD3 Networks
        self.actor = Actor(self.input_dim, self.action_dim, self.max_action)
        self.actor_target = Actor(self.input_dim, self.action_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = Critic(self.input_dim, self.action_dim)
        self.critic_target = Critic(self.input_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=5000, gamma=0.9)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=5000, gamma=0.9)

        self.memory = deque(maxlen=50000)
        
        # Prioritized Replay: separate buffer for high-reward episodes
        self.priority_memory = deque(maxlen=10000)  # Store successful episodes
        self.current_episode_buffer = []  # Temporary buffer for current episode
        self.episode_scores = deque(maxlen=100)  # Track recent episode scores
        
        self.steps = 0
        self.total_steps = 0  # For delayed policy updates
        self.expl_noise = EXPL_NOISE  # Exploration noise for continuous actions
        self.consecutive_crashes = 0  # Track consecutive crashes for early stopping
        
        # Locations
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200)  # Current active target
        
        # Multiple Targets Support
        self.targets = []  # List of target positions
        self.current_target_idx = 0  # Index of current active target
        self.targets_reached = 0  # Counter for completed targets
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None
        
        # Tracking for UI display
        self.best_score = float('-inf')
        self.successes = 0  # Number of successful episodes
        self.current_steering = 0.0
        self.current_throttle = MIN_THROTTLE
        self.current_speed = 0.0  # Current speed in px/step
        self.avg_q_score = 0.0  # Average Q-value
        self.q_score_history = deque(maxlen=100)  # Track Q-values for averaging
        
        # Learning metrics
        self.episode_lengths = deque(maxlen=100)  # Track episode lengths
        self.avg_episode_reward = 0.0  # Average reward per episode
        self.avg_episode_reward_history = deque(maxlen=100)  # Track average rewards
        self.crashes = 0  # Total crashes
        self.total_episodes = 0  # Total episodes
        self.avg_distance_to_target = 0.0  # Average distance to target
        self.distance_history = deque(maxlen=1000)  # Track distances
        self.q_score_trend = 0.0  # Q-score improvement trend (positive = improving)
        
        # Track recent states to avoid repeating mistakes
        self.recent_states = deque(maxlen=50)  # Track last 50 states
        self.recent_actions = deque(maxlen=50)  # Track last 50 actions
        self.repeat_penalty_factor = 1.0  # Penalty multiplier for repeated patterns

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        # Reset to first target
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        # Reset progress history to track circling
        self.progress_history = deque(maxlen=30)
        # Reset recent states/actions to avoid false positives
        if hasattr(self, 'recent_states'):
            self.recent_states.clear()
            self.recent_actions.clear()
        return state
    
    def add_target(self, point):
        """Add a new target to the sequence"""
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            # First target, set it as active
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        """Switch to the next target in sequence"""
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True  # More targets available
        return False  # All targets completed

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        # 7 sensors: -45°, -30°, -15°, 0°, 15°, 30°, 45°
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos.x() + math.cos(rad) * SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))
            
            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)
            
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180: angle_diff -= 360
        
        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        # TD3: action is now [steering, throttle]
        # action[0] = steering angle (degrees)
        # action[1] = throttle multiplier (speed factor)

        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Ensure action has 2 elements
        if len(action) != 2:
            print(f"ERROR in step(): Action has {len(action)} elements, expected 2")
            action = np.array([0.0, MIN_THROTTLE])

        # Extract and clamp each action dimension separately
        steering = float(np.clip(action[0], -MAX_STEERING, MAX_STEERING))
        throttle = float(np.clip(action[1], MIN_THROTTLE, MAX_THROTTLE))
        
        # Safety check: ensure throttle is at least MIN_THROTTLE
        if throttle < MIN_THROTTLE:
            throttle = MIN_THROTTLE
        
        # Store current actions and speed for UI display
        self.current_steering = steering
        self.current_throttle = throttle
        self.current_speed = SPEED * throttle  # Calculate current speed

        # Debug: Print step details for first few steps
        if self.steps < 5:
            print(f"  Step {self.steps}: steering={steering:.2f}°, throttle={throttle:.2f}, "
                  f"speed={SPEED * throttle:.2f}, pos=({self.car_pos.x():.1f}, {self.car_pos.y():.1f})")

        # Apply steering
        self.car_angle += steering
        rad = math.radians(self.car_angle)
        
        # Apply throttle to speed
        current_speed = SPEED * throttle

        # Update position
        old_x, old_y = self.car_pos.x(), self.car_pos.y()
        new_x = old_x + math.cos(rad) * current_speed
        new_y = old_y + math.sin(rad) * current_speed
        self.car_pos = QPointF(new_x, new_y)
        
        # Debug: Verify position actually changed
        if self.steps < 5:
            print(f"  Position update: ({old_x:.1f}, {old_y:.1f}) -> ({new_x:.1f}, {new_y:.1f}), "
                  f"delta=({new_x-old_x:.3f}, {new_y-old_y:.3f})")
        
        next_state, dist = self.get_state()
        sensors = next_state[:7]  # 7 sensor readings
        
        # Track distance for learning metrics
        self.distance_history.append(dist)
        if len(self.distance_history) > 0:
            self.avg_distance_to_target = sum(self.distance_history) / len(self.distance_history)
        
        reward = -0.1 # Base step penalty
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        # CRASH DETECTION: Only check if CAR is off the road (not sensors)
        # Sensors provide information for learning, but don't trigger crashes
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        elif dist < 20: 
            # Target reached!
            reward = 100
            # Check if there are more targets
            has_next = self.switch_to_next_target()
            if has_next:
                # Continue to next target
                done = False
                # Reset distance tracking for new target
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                # All targets completed
                done = True
        else:
            # Improved reward shaping to prevent circling and repeating mistakes:
        
            # 1. Reward for staying on road (center sensor)
            center_sensor = next_state[3]  # 0° sensor (forward)
            reward += center_sensor * 3.0  # Bonus for seeing road ahead

            # 2. Strong reward for progress toward target (most important!)
            progress = 0.0
            if self.prev_dist is not None:
                progress = self.prev_dist - dist
                # Strong reward for getting closer (scaled by distance)
                progress_reward = progress * 8.0  # Increased to emphasize progress
                reward += progress_reward
                
                # Strong penalty for moving away
                if dist > self.prev_dist:
                    reward -= 30  # Increased penalty
                
                # Bonus for making significant progress
                if progress > 3.0:  # Moved more than 3 pixels closer
                    reward += 3.0

            # 3. Penalty for excessive steering (prevents circling)
            abs_steering = abs(steering)
            # Smooth penalty that increases with steering angle
            steering_penalty = (abs_steering / MAX_STEERING) * 1.0
            reward -= steering_penalty
            
            # 4. Reward for maintaining forward direction toward target
            angle_to_target = next_state[7]  # Normalized angle difference [-1, 1]
            # Reward for pointing toward target (small angle difference)
            direction_reward = (1.0 - abs(angle_to_target)) * 3.0
            reward += direction_reward

            # 5. Reward for maintaining reasonable speed
            reward += throttle * 0.3  # Small bonus for speed
            
            # 6. Penalty for repeating similar states/actions (avoid repeating mistakes)
            state_key = tuple(np.round(next_state[:5], 2))  # Use first 5 state values as key
            action_key = (round(steering, 1), round(throttle, 1))
            
            # Check if we've seen similar state-action pairs recently
            repeat_count = 0
            for i, (past_state, past_action) in enumerate(zip(self.recent_states, self.recent_actions)):
                # Check if state is similar (within threshold)
                state_diff = sum(abs(s1 - s2) for s1, s2 in zip(state_key, past_state))
                action_diff = abs(action_key[0] - past_action[0]) + abs(action_key[1] - past_action[1])
                
                if state_diff < 0.3 and action_diff < 0.5:  # Similar state and action
                    repeat_count += 1
            
            # Penalize repeating the same mistakes
            if repeat_count > 2:  # If we've done this 3+ times recently
                repeat_penalty = repeat_count * 2.0
                reward -= repeat_penalty
            
            # Store current state and action for future comparison
            self.recent_states.append(state_key)
            self.recent_actions.append(action_key)
            
            # 7. Penalty for not making progress over multiple steps
            if not hasattr(self, 'progress_history'):
                self.progress_history = deque(maxlen=30)  # Last 30 steps
            if self.prev_dist is not None:
                self.progress_history.append(progress)
                # If average progress is negative over last steps, penalize heavily
                if len(self.progress_history) >= 15:
                    avg_progress = sum(self.progress_history) / len(self.progress_history)
                    if avg_progress < -0.5:  # Consistently moving away or stuck
                        reward -= 10.0  # Strong penalty for circling/stuck behavior
            
            # 8. Reward for balanced sensors (not too close to walls)
            left_sensors = (next_state[0] + next_state[1] + next_state[2]) / 3.0
            right_sensors = (next_state[4] + next_state[5] + next_state[6]) / 3.0
            sensor_balance = min(left_sensors, right_sensors)
            reward += sensor_balance * 1.5  # Bonus for balanced path
            
            self.prev_dist = dist
            
        self.score += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def optimize(self):
        """TD3 Training: Twin Critic updates + Delayed Actor updates"""
        total_memory_size = len(self.memory) + len(self.priority_memory)
        if total_memory_size < BATCH_SIZE: return 0
        
        # ADAPTIVE PRIORITIZED SAMPLING with diversity
        success_rate = len(self.priority_memory) / max(total_memory_size, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)
        priority_samples = int(BATCH_SIZE * priority_ratio)
        regular_samples = BATCH_SIZE - priority_samples
        
        batch = []
        
        # Sample from priority memory with diversity (avoid sampling too similar experiences)
        if len(self.priority_memory) >= priority_samples:
            # Use random sampling but ensure diversity by avoiding consecutive similar samples
            sampled = random.sample(self.priority_memory, min(priority_samples, len(self.priority_memory)))
            batch.extend(sampled)
        else:
            batch.extend(list(self.priority_memory))
            regular_samples += priority_samples - len(self.priority_memory)
        
        # Sample from regular memory (include mistakes to learn from them)
        if len(self.memory) >= regular_samples:
            # Ensure we sample diverse experiences, not just recent ones
            sampled = random.sample(self.memory, regular_samples)
            batch.extend(sampled)
        else:
            batch.extend(list(self.memory))
        
        if len(batch) < BATCH_SIZE // 2:
            return 0
        
        # Unpack batch
        s, a, r, ns, d = zip(*batch)
        state = torch.FloatTensor(np.array(s))
        action = torch.FloatTensor(np.array(a)) # Shape: (batch_size, 2) - Remove .unsqueeze(1)
        reward = torch.FloatTensor(r).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(ns))
        done = torch.FloatTensor(d).unsqueeze(1)
        
        # ===== CRITIC UPDATE =====
        # TD3: Select continuous action from actor [steering, throttle]
        with torch.no_grad():
            # Compute next_action from target actor (for target Q-value)
            next_action = self.actor_target(next_state)

            # Add target policy smoothing noise (TD3 key feature)
            noise = torch.randn_like(next_action) * POLICY_NOISE
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
            next_action = next_action + noise

            # Clip each action dimension separately
            next_action[:, 0] = torch.clamp(next_action[:, 0], -MAX_STEERING, MAX_STEERING)
            next_action[:, 1] = torch.clamp(next_action[:, 1], MIN_THROTTLE, MAX_THROTTLE)  # Use MIN_THROTTLE
            
            # Compute target Q-values using twin critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)  # Take minimum (clipped double Q-learning)
            target_Q = reward + (1 - done) * GAMMA * target_Q
        
        # Get current Q-values
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Track Q-score for UI (average of Q1 and Q2)
        avg_q = (current_Q1.mean().item() + current_Q2.mean().item()) / 2.0
        self.q_score_history.append(avg_q)
        self.avg_q_score = sum(self.q_score_history) / len(self.q_score_history) if self.q_score_history else 0.0
        
        # Calculate Q-score trend (improving = less negative or more positive)
        if len(self.q_score_history) >= 20:
            recent_avg = sum(list(self.q_score_history)[-20:]) / 20
            older_avg = sum(list(self.q_score_history)[-40:-20]) / 20 if len(self.q_score_history) >= 40 else recent_avg
            self.q_score_trend = recent_avg - older_avg  # Positive = improving
        
        # Calculate Q-score trend (improving = less negative or more positive)
        if len(self.q_score_history) >= 20:
            recent_avg = sum(list(self.q_score_history)[-20:]) / 20
            older_avg = sum(list(self.q_score_history)[-40:-20]) / 20 if len(self.q_score_history) >= 40 else recent_avg
            self.q_score_trend = recent_avg - older_avg  # Positive = improving
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # if self.total_steps % 100 == 0:  # Update every 100 steps
        #     self.actor_scheduler.step()
        #     self.critic_scheduler.step()
        
        # ===== DELAYED ACTOR UPDATE (TD3 key feature) =====
        actor_loss = 0
        if self.total_steps % POLICY_FREQ == 0:
            # Compute actor loss: maximize Q1(s, actor(s))
            actor_action = self.actor(state)
            actor_loss = -self.critic.Q1(state, actor_action).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks (Polyak averaging)
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        
        self.total_steps += 1
        
        # Adaptive exploration noise decay - slower decay to maintain exploration
        # Decay more slowly to avoid getting stuck in local minima and repeating mistakes
        if self.expl_noise > 0.05:
            # Slower decay: 0.9998 instead of 0.9995 to maintain exploration longer
            self.expl_noise *= 0.9998
        elif self.expl_noise > 0.01:
            # Even slower decay when noise is low to prevent premature convergence
            self.expl_noise *= 0.9999
        
        return critic_loss.item() + (actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0)
    
    def store_experience(self, experience):
        """Store experience in current episode buffer"""
        self.current_episode_buffer.append(experience)
    
    def finalize_episode(self, episode_reward):
        """Move episode experiences to appropriate memory based on reward"""
        if len(self.current_episode_buffer) == 0:
            return
        
        # Track episode score
        self.episode_scores.append(episode_reward)
        
        # Track learning metrics
        self.total_episodes += 1
        self.episode_lengths.append(self.steps)
        self.avg_episode_reward_history.append(episode_reward)
        self.avg_episode_reward = sum(self.avg_episode_reward_history) / len(self.avg_episode_reward_history) if self.avg_episode_reward_history else 0.0
        
        # Track consecutive crashes for early stopping
        if not self.alive:  # Episode ended in crash
            self.consecutive_crashes += 1
            self.crashes += 1
        else:  # Episode ended successfully
            self.consecutive_crashes = 0  # Reset counter on success
        
        # Determine if this is a high-reward episode
        # High reward = positive score (successful navigation)
        if episode_reward > 0:
            # Add to priority memory (high-reward episodes)
            for exp in self.current_episode_buffer:
                self.priority_memory.append(exp)
        else:
            # Add to regular memory
            for exp in self.current_episode_buffer:
                self.memory.append(exp)
        
        # Clear episode buffer for next episode
        self.current_episode_buffer = []

# ==========================================
# 4. CUSTOM WIDGETS (VISUALS)
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50 # How many episodes to show

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        if len(self.scores) < 2:
            return

        # Normalize data
        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: max_val += 1
        
        # Calculate points
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, score in enumerate(self.scores):
            x = i * step_x
            # Normalize y to 10% padding top/bottom
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        # Draw Path (Raw scores)
        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        # Draw Raw Score Line
        pen = QPen(C_ACCENT, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Calculate and draw moving average (last 10)
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                # Calculate average of last 'window_size' scores up to current index
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            # Draw moving average line
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                # Draw with yellow/gold color and thicker line
                avg_pen = QPen(QColor(255, 215, 0), 3)  # Gold color
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        # Draw Zero Line if visible
        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))
        
        # Draw Legend
        legend_x = 10
        legend_y = 15
        
        # Raw score legend
        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")
        
        # Moving average legend
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")

class SensorItem(QGraphicsItem):
    """Animated sensor dot with pulsing effect"""
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True  # Whether sensor sees road
        
    def set_detecting(self, detecting):
        """Update sensor state (road/obstacle)"""
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        # Pulsing animation
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        # Color based on detection
        if self.is_detecting:
            color = C_SENSOR_ON  # Green - sees road
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF  # Red - sees obstacle
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Outer glow (pulsing)
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        # Inner core (solid)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active  # Active target pulses, inactive is dimmed
        self.number = number  # Sequence number (1, 2, 3, ...)

    def set_active(self, active):
        """Set whether this target is currently active"""
        self.is_active = active
        self.update()
    
    def set_color(self, color):
        """Update the color of this target"""
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            # Active target pulses
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: self.growing = True
            
            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0,0), r, r)
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0,0), 8, 8)
        else:
            # Inactive target is smaller and dimmed
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0,0), 6, 6)
        
        # Draw sequence number
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 5. APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav: Precise Control")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        # Header
        lbl_title = QLabel("CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)
        
        # Status Box
        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        # Buttons
        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)

        # Chart Section
        vbox.addSpacing(15)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        # Stats Grid (Expanded with all requested metrics)
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        sf_layout.setSpacing(5)
        
        row = 0
        # Q-Score
        self.val_qscore = QLabel("0.00")
        self.val_qscore.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Q-Score:"), row, 0)
        sf_layout.addWidget(self.val_qscore, row, 1)
        row += 1
        
        # Best Score
        self.val_best_score = QLabel("0")
        self.val_best_score.setStyleSheet(f"color: {C_SUCCESS.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Best Score:"), row, 0)
        sf_layout.addWidget(self.val_best_score, row, 1)
        row += 1
        
        # Steps
        self.val_steps = QLabel("0")
        self.val_steps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Steps:"), row, 0)
        sf_layout.addWidget(self.val_steps, row, 1)
        row += 1
        
        # Learning Rate (Actor)
        self.val_lr_actor = QLabel(f"{LR_ACTOR:.6f}")
        self.val_lr_actor.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("LR (Actor):"), row, 0)
        sf_layout.addWidget(self.val_lr_actor, row, 1)
        row += 1
        
        # Learning Rate (Critic)
        self.val_lr_critic = QLabel(f"{LR_CRITIC:.6f}")
        self.val_lr_critic.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("LR (Critic):"), row, 0)
        sf_layout.addWidget(self.val_lr_critic, row, 1)
        row += 1
        
        # Current Steering
        self.val_steering = QLabel("0.0°")
        self.val_steering.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Steering:"), row, 0)
        sf_layout.addWidget(self.val_steering, row, 1)
        row += 1
        
        # Current Throttle
        self.val_throttle = QLabel("0.0")
        self.val_throttle.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Throttle:"), row, 0)
        sf_layout.addWidget(self.val_throttle, row, 1)
        row += 1
        
        # Current Speed
        self.val_speed = QLabel("0.0 px/s")
        self.val_speed.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Speed:"), row, 0)
        sf_layout.addWidget(self.val_speed, row, 1)
        row += 1
        
        # Successes
        self.val_successes = QLabel("0")
        self.val_successes.setStyleSheet(f"color: {C_SUCCESS.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Successes:"), row, 0)
        sf_layout.addWidget(self.val_successes, row, 1)
        row += 1
        
        # Time Elapsed
        self.val_time = QLabel("0:00")
        self.val_time.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Time:"), row, 0)
        sf_layout.addWidget(self.val_time, row, 1)
        row += 1
        
        # Epsilon Noise (Exploration Noise)
        self.val_eps = QLabel("0.10")
        self.val_eps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Epsilon Noise:"), row, 0)
        sf_layout.addWidget(self.val_eps, row, 1)
        row += 1
        
        # Last Reward
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Last Reward:"), row, 0)
        sf_layout.addWidget(self.val_rew, row, 1)
        row += 1
        
        # Average Episode Reward (learning indicator)
        self.val_avg_reward = QLabel("0.0")
        self.val_avg_reward.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Avg Reward:"), row, 0)
        sf_layout.addWidget(self.val_avg_reward, row, 1)
        row += 1
        
        # Average Episode Length
        self.val_avg_length = QLabel("0")
        self.val_avg_length.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Avg Length:"), row, 0)
        sf_layout.addWidget(self.val_avg_length, row, 1)
        row += 1
        
        # Crash Rate
        self.val_crash_rate = QLabel("0%")
        self.val_crash_rate.setStyleSheet(f"color: {C_FAILURE.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Crash Rate:"), row, 0)
        sf_layout.addWidget(self.val_crash_rate, row, 1)
        row += 1
        
        # Avg Distance to Target
        self.val_avg_dist = QLabel("0.0")
        self.val_avg_dist.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Avg Distance:"), row, 0)
        sf_layout.addWidget(self.val_avg_dist, row, 1)
        row += 1
        
        # Q-Score Trend (Learning Indicator)
        self.val_q_trend = QLabel("→")
        self.val_q_trend.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold; font-size: 16px;")
        sf_layout.addWidget(QLabel("Q Trend:"), row, 0)
        sf_layout.addWidget(self.val_q_trend, row, 1)
        
        vbox.addWidget(stats_frame)
        
        # Save/Load Weights Buttons
        weights_frame = QFrame()
        weights_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        wf_layout = QVBoxLayout(weights_frame)
        wf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.btn_save_weights = QPushButton("💾 SAVE BEST WEIGHTS")
        self.btn_save_weights.clicked.connect(self.save_best_weights)
        wf_layout.addWidget(self.btn_save_weights)
        
        self.btn_load_weights = QPushButton("📂 LOAD WEIGHTS")
        self.btn_load_weights.clicked.connect(self.load_weights)
        wf_layout.addWidget(self.btn_load_weights)
        
        vbox.addWidget(weights_frame)

        # Logs
        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Logic
        self.setup_map("city_map.png") 
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        # Time tracking
        self.start_time = None
        self.best_weights_path = "best_weights.pth"
        
        self.car_item = CarItem()
        self.target_items = []  # List of target items
        self.sensor_items = []
        # Create 7 animated sensors
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded.")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: 
            self.full_reset()
            self.setup_map(f)

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            # Set car position
            self.brain.set_start_pos(pt) 
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map -> TARGET(S)\nRight-click when done")
        elif self.setup_state == 1:
            # Add target
            if event.button() == Qt.MouseButton.LeftButton:
                # Add a new target
                self.brain.add_target(pt)
                target_idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)  # First target is active
                num_targets = len(self.brain.targets)
                
                target_item = TargetItem(color, is_active, num_targets)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish setup")
                self.log(f"Target #{num_targets} added at ({pt.x():.0f}, {pt.y():.0f})")
            
            elif event.button() == Qt.MouseButton.RightButton:
                # Finish setup if at least one target exists
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        self.scene.removeItem(self.car_item)
        # Remove all target items
        for target_item in self.target_items:
            self.scene.removeItem(target_item)
        self.target_items = []
        # Clear brain targets
        self.brain.targets = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        
        for s in self.sensor_items: 
            if s.scene() == self.scene: self.scene.removeItem(s)
        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("--- RESET ---")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            if self.start_time is None:
                self.start_time = time.time()
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")
    
    def save_best_weights(self):
        """Save the best weights to file"""
        try:
            checkpoint = {
                'actor_state_dict': self.brain.actor.state_dict(),
                'actor_target_state_dict': self.brain.actor_target.state_dict(),
                'critic_state_dict': self.brain.critic.state_dict(),
                'critic_target_state_dict': self.brain.critic_target.state_dict(),
                'actor_optimizer': self.brain.actor_optimizer.state_dict(),
                'critic_optimizer': self.brain.critic_optimizer.state_dict(),
                'best_score': self.brain.best_score,
                'total_steps': self.brain.total_steps,
                'successes': self.brain.successes,
            }
            torch.save(checkpoint, self.best_weights_path)
            self.log(f"<font color='{C_SUCCESS.name()}'>✅ Best weights saved to {self.best_weights_path}</font>")
        except Exception as e:
            self.log(f"<font color='{C_FAILURE.name()}'>❌ Error saving weights: {str(e)}</font>")
    
    def load_weights(self):
        """Load weights from file"""
        try:
            f, _ = QFileDialog.getOpenFileName(self, "Load Weights", "", "PyTorch Files (*.pth)")
            if f:
                checkpoint = torch.load(f)
                self.brain.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.brain.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                self.brain.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.brain.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                self.brain.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.brain.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                if 'best_score' in checkpoint:
                    self.brain.best_score = checkpoint['best_score']
                if 'successes' in checkpoint:
                    self.brain.successes = checkpoint['successes']
                self.log(f"<font color='{C_SUCCESS.name()}'>✅ Weights loaded from {f}</font>")
                self.log(f"<font color='{C_ACCENT.name()}'>Best Score: {self.brain.best_score}, Successes: {self.brain.successes}</font>")
        except Exception as e:
            self.log(f"<font color='{C_FAILURE.name()}'>❌ Error loading weights: {str(e)}</font>")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        if self.setup_state != 2: return

        state, _ = self.brain.get_state()
        
        # Track current target index before step
        prev_target_idx = self.brain.current_target_idx
        
        # TD3: Select continuous action from actor
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = self.brain.actor(state_tensor)
            action = action_tensor.cpu().numpy().flatten()
            
            # Debug: Print action values for first few steps
            if self.brain.steps < 5:
                print(f"Step {self.brain.steps}: Raw actor output: {action_tensor.cpu().numpy()}, Flattened: {action}")
            
            # Ensure action is a valid 1D array with 2 elements
            if len(action) != 2:
                print(f"ERROR: Invalid action shape: {action.shape}, expected (2,)")
                action = np.array([0.0, MIN_THROTTLE])
            
            # Add exploration noise (TD3 exploration strategy)
            if self.brain.expl_noise > 0:
                noise = np.random.normal(0, self.brain.expl_noise, size=action.shape)
                action = action + noise
            
            # ALWAYS clip actions to valid bounds (critical fix!)
            action[0] = np.clip(action[0], -MAX_STEERING, MAX_STEERING)
            action[1] = np.clip(action[1], MIN_THROTTLE, MAX_THROTTLE)
            
            # Debug: Print final action values
            if self.brain.steps < 10 or self.brain.steps % 50 == 0:
                print(f"Step {self.brain.steps}: Final action=[steering: {action[0]:.2f}°, throttle: {action[1]:.2f}], "
                      f"Speed={SPEED * action[1]:.2f} px/step, ExplNoise={self.brain.expl_noise:.3f}")

        next_s, rew, done = self.brain.step(action)
        
        # Store experience in current episode buffer (action is now continuous)
        self.brain.store_experience((state, action, rew, next_s, done))
        
        # Update visuals BEFORE early return (critical fix!)
        self.update_visuals()
        
        # Increment step counters
        self.brain.steps += 1
        self.brain.total_steps += 1
        
        # Update all UI metrics
        self.update_ui_metrics()
        
        if self.brain.total_steps < 1000:  # Collect experiences first
            # Don't optimize, just collect
            return
        
        self.brain.optimize()
        
        # Check if target switched
        if self.brain.current_target_idx != prev_target_idx:
            target_num = self.brain.current_target_idx + 1
            total = len(self.brain.targets)
            self.log(f"<font color='#88C0D0'>🎯 Target {prev_target_idx + 1} reached! Moving to target {target_num}/{total}</font>")
        
        if done:
            # Finalize episode and move experiences to appropriate memory
            self.brain.finalize_episode(self.brain.score)
            
            # Determine reset behavior based on crash counter
            should_reset_position = False
            
            # Check for max consecutive crashes (reset to origin)
            if self.brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='#BF616A'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting to origin...</b></font>")
                self.log(f"<font color='#88C0D0'>💡 Tip: Adjust hyperparameters, simplify map, or increase exploration noise</font>")
                # Reset crash counter and position
                self.brain.consecutive_crashes = 0
                should_reset_position = True
            
            if not self.brain.alive:
                txt = f"CRASH ({self.brain.consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES})"
                col = "#BF616A"
            else:
                # Check if all targets completed
                if self.brain.targets_reached == len(self.brain.targets) - 1:
                    txt = f"ALL {len(self.brain.targets)} TARGETS COMPLETED!"
                    col = "#A3BE8C"
                    # Count as success
                    self.brain.successes += 1
                else:
                    txt = "GOAL"
                    col = "#A3BE8C"
                    # Count as success
                    self.brain.successes += 1
                # Always reset position on success
                should_reset_position = True
            
            # Update best score
            if self.brain.score > self.brain.best_score:
                self.brain.best_score = self.brain.score
                # Auto-save best weights if score improved significantly
                if self.brain.score > 50:  # Only save if score is meaningful
                    try:
                        checkpoint = {
                            'actor_state_dict': self.brain.actor.state_dict(),
                            'actor_target_state_dict': self.brain.actor_target.state_dict(),
                            'critic_state_dict': self.brain.critic.state_dict(),
                            'critic_target_state_dict': self.brain.critic_target.state_dict(),
                            'actor_optimizer': self.brain.actor_optimizer.state_dict(),
                            'critic_optimizer': self.brain.critic_optimizer.state_dict(),
                            'best_score': self.brain.best_score,
                            'total_steps': self.brain.total_steps,
                            'successes': self.brain.successes,
                        }
                        torch.save(checkpoint, self.best_weights_path)
                    except Exception as e:
                        pass  # Silent fail for auto-save
            
            # Log memory statistics with adaptive sampling ratio
            priority_size = len(self.brain.priority_memory)
            regular_size = len(self.brain.memory)
            total_mem = priority_size + regular_size
            priority_pct = (priority_size / total_mem * 100) if total_mem > 0 else 0
            
            # Calculate current adaptive sampling ratio
            success_rate = priority_size / max(total_mem, 1)
            sampling_ratio = 0.3 + (success_rate * 0.4)
            
            self.log(f"<font color='{col}'>{txt} (Scr: {self.brain.score:.0f}) | "
                    f"Mem: {priority_size}P/{regular_size}R ({priority_pct:.1f}%) | "
                    f"Sample: {sampling_ratio*100:.0f}%P</font>")
            
            # Update Chart
            self.chart.update_chart(self.brain.score)
            
            # Reset based on condition
            if should_reset_position:
                self.brain.reset()  # Full reset (position + score + targets)
            else:
                # Partial reset: keep position, reset score and targets
                # Car stays where it crashed but gets fresh attempt
                self.brain.score = 0
                self.brain.alive = True
                self.brain.current_target_idx = 0
                self.brain.targets_reached = 0
                if len(self.brain.targets) > 0:
                    self.brain.target_pos = self.brain.targets[0]
                # Reset distance tracking
                _, dist = self.brain.get_state()
                self.brain.prev_dist = dist
                # Update visuals after reset
                self.update_visuals()
        
        # Update UI metrics (called every frame)
        self.update_ui_metrics()
    
    def update_ui_metrics(self):
        """Update all UI metric displays"""
        # Q-Score (can be negative - that's normal with negative rewards)
        # Color code: green if > -50, red if worse
        q_color = C_SUCCESS.name() if self.brain.avg_q_score > -50 else C_FAILURE.name()
        self.val_qscore.setStyleSheet(f"color: {q_color}; font-weight: bold;")
        self.val_qscore.setText(f"{self.brain.avg_q_score:.2f}")
        
        # Q-Score Trend (Learning Indicator) - shows if Q is improving
        if abs(self.brain.q_score_trend) < 0.1:
            trend_symbol = "→"  # Stable
            trend_color = C_ACCENT.name()
        elif self.brain.q_score_trend > 0:
            trend_symbol = "↑"  # Improving (less negative or more positive)
            trend_color = C_SUCCESS.name()
        else:
            trend_symbol = "↓"  # Declining
            trend_color = C_FAILURE.name()
        self.val_q_trend.setText(trend_symbol)
        self.val_q_trend.setStyleSheet(f"color: {trend_color}; font-weight: bold; font-size: 16px;")
        
        # Best Score
        self.val_best_score.setText(f"{self.brain.best_score:.0f}")
        
        # Steps
        self.val_steps.setText(f"{self.brain.total_steps}")
        
        # Learning Rates (get from optimizers)
        actor_lr = self.brain.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.brain.critic_optimizer.param_groups[0]['lr']
        self.val_lr_actor.setText(f"{actor_lr:.6f}")
        self.val_lr_critic.setText(f"{critic_lr:.6f}")
        
        # Current Steering, Throttle, and Speed
        self.val_steering.setText(f"{self.brain.current_steering:.1f}°")
        self.val_throttle.setText(f"{self.brain.current_throttle:.2f}")
        self.val_speed.setText(f"{self.brain.current_speed:.2f} px/s")
        
        # Successes
        self.val_successes.setText(f"{self.brain.successes}")
        
        # Average Episode Reward (KEY LEARNING METRIC - should increase over time)
        self.val_avg_reward.setText(f"{self.brain.avg_episode_reward:.1f}")
        
        # Average Episode Length (should increase if learning)
        avg_len = sum(self.brain.episode_lengths) / len(self.brain.episode_lengths) if self.brain.episode_lengths else 0
        self.val_avg_length.setText(f"{avg_len:.0f}")
        
        # Crash Rate (should decrease if learning)
        crash_rate = (self.brain.crashes / self.brain.total_episodes * 100) if self.brain.total_episodes > 0 else 0
        self.val_crash_rate.setText(f"{crash_rate:.1f}%")
        
        # Average Distance to Target (should decrease if learning)
        self.val_avg_dist.setText(f"{self.brain.avg_distance_to_target:.1f}")
        
        # Time Elapsed
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.val_time.setText(f"{minutes}:{seconds:02d}")
        else:
            self.val_time.setText("0:00")
        
        # Epsilon Noise (Exploration Noise)
        self.val_eps.setText(f"{self.brain.expl_noise:.3f}")
        
        # Last Reward (current score)
        self.val_rew.setText(f"{self.brain.score:.0f}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        # Update target visuals based on current active target
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target_item.set_active(is_active)
        
        self.scene.update() 
        
        # Update all 7 sensors
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            s_val = self.brain.get_state()[0][i]
            self.sensor_items[i].set_detecting(s_val > 0.5)  # True if sees road

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec())