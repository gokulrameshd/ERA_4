# Autonomous Car Navigation using Deep Q-Network (DQN)

## Overview

This project implements a self-driving car navigation system using Deep Q-Network (DQN) reinforcement learning. The car learns to navigate through a city map, avoiding obstacles and reaching target destinations by learning from trial and error.

## Table of Contents

1. [Architecture](#architecture)
2. [Hyperparameters](#hyperparameters)
3. [Reward Structure](#reward-structure)
4. [Training Strategies](#training-strategies)
5. [Key Features](#key-features)
6. [Usage](#usage)
7. [Monitoring Training](#monitoring-training)
8. [Troubleshooting](#troubleshooting)

---

## Architecture

### Neural Network

**Network Type:** Deep Q-Network (DQN) with Target Network

**Architecture:**
```
Input Layer:  9 neurons (7 sensors + angle_to_target + distance_to_target)
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2: 256 neurons (ReLU)
Hidden Layer 3: 256 neurons (ReLU)
Hidden Layer 4: 128 neurons (ReLU)
Output Layer: 5 neurons (one for each action)
```

**Actions:**
- 0: Left turn
- 1: Straight
- 2: Right turn
- 3: Sharp left turn
- 4: Sharp right turn

### State Representation

The agent observes:
- **7 Sensor Values:** Distance sensors at angles [-45°, -30°, -15°, 0°, 15°, 30°, 45°]
  - Each sensor returns brightness value (0.0 = off-road, 1.0 = road)
- **Angle to Target:** Normalized angle difference (-1.0 to 1.0)
- **Distance to Target:** Normalized distance (0.0 to 1.0)

---

## Hyperparameters

### Physics Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `SENSOR_DIST` | 15 pixels | Short sensors for narrow roads to avoid false positives from obstacles on opposite side |
| `SPEED` | 2.0 pixels/step | Moderate speed for balance between progress and control |
| `TURN_SPEED` | 1.0°/step | Good responsiveness for both gentle curves and moderate turns |
| `SHARP_TURN` | 18° | Handles tight corners (90°+ turns) effectively |
| `CAR_WIDTH` | 14 pixels | Physical car dimensions |
| `CAR_HEIGHT` | 8 pixels | Physical car dimensions |

**Rationale:**
- Short sensor distance (15px) is chosen because the city map has narrow roads. Longer sensors would detect obstacles on the opposite side of narrow streets, causing confusion.
- Speed of 2.0 provides reaction time of ~7.5 steps (15/2), which is adequate for the sensor range.
- Turn speeds are balanced to handle both gentle curves and sharp corners.

### Reinforcement Learning Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `BATCH_SIZE` | 128 | Smaller batches for more stable learning, reduces variance |
| `GAMMA` | 0.99 | High discount factor for long-term planning (essential for navigation) |
| `LR` | 0.00005 | Very low learning rate to prevent Q-value explosion and ensure stable learning |
| `TAU` | 0.01 | Faster target network updates for more stable Q-value estimates |
| `EPSILON_MIN` | 0.001 | Minimal exploration once learned, allows pure exploitation |
| `MAX_CONSECUTIVE_CRASHES` | 2 | Quick reset after crashes to avoid wasting time on bad configurations |

**Rationale:**
- **Low Learning Rate (0.00005):** Prevents Q-value explosion. Higher rates caused loss values > 1000 and unstable learning.
- **High Gamma (0.99):** Navigation requires long-term planning. The agent must consider consequences 20-100 steps ahead.
- **Small Batch Size (128):** Reduces variance in gradient estimates, leading to more stable updates.
- **Fast Target Updates (TAU=0.01):** Helps stabilize Q-value estimates by updating target network more frequently.

### Memory Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `memory` | 20,000 | Large buffer for diverse experiences |
| `priority_memory` | 10,000 | Stores successful episodes for prioritized replay |
| `episode_scores` | 200 | Tracks recent performance for monitoring |

**Rationale:**
- Larger memory buffers provide more diverse training samples
- Separate priority memory ensures successful trajectories are sampled more frequently

### Training Stabilization

| Technique | Value | Reason |
|-----------|-------|--------|
| **Gradient Clipping** | max_norm=0.5 | Prevents gradient explosion, critical for stability |
| **Q-Value Clipping** | [-200, 200] | Prevents Q-value explosion, keeps values in reasonable range |

**Rationale:**
- Gradient clipping prevents large gradients from destabilizing learning
- Q-value clipping directly addresses the problem of Q-scores growing too high (>500)

---

## Reward Structure

### Progressive Boundary Penalties

The reward system uses **progressive penalties** based on how close the car is to boundaries:

```python
if car_center_val < 0.4:    # Off-road
    reward = -300
elif car_center_val < 0.5:   # Near boundary
    reward = -150
elif car_center_val < 0.6:   # Approaching boundary
    reward = -75
```

**Strategy:** Progressive penalties teach the agent to avoid boundaries gradually, not just when crashing. This provides a stronger learning signal than binary crash/no-crash.

### Target Rewards

```python
if dist < 25:  # Reached target
    reward = 200
```

**Strategy:** High positive reward (200) creates strong incentive to reach targets. The ratio of crash penalty (-300) to target reward (200) is 1.5:1, ensuring the agent prioritizes avoiding crashes.

### Shaping Rewards

```python
reward = -0.05  # Step penalty (time cost)
reward += (1.0 - next_state[4]) * 20  # Orientation bonus (facing target)
if dist > prev_dist:
    reward -= 8  # Distance penalty (moving away)
```

**Strategy:**
- Small step penalty (-0.05) encourages efficiency without being too harsh
- Orientation bonus rewards facing the target, providing guidance
- Distance penalty discourages moving away from target

### Reward Ratio Analysis

| Event | Reward | Ratio to Target |
|-------|--------|-----------------|
| Crash (off-road) | -300 | -1.5x |
| Near boundary | -150 | -0.75x |
| Approaching boundary | -75 | -0.375x |
| Reached target | +200 | 1.0x |
| Step penalty | -0.05 | Negligible |
| Orientation bonus | +0 to +20 | 0 to 0.1x |

**Rationale:** The strong negative signal for crashes ensures the agent learns to avoid boundaries as the highest priority.

---

## Training Strategies

### 1. Prioritized Experience Replay

**Implementation:**
- Separate memory buffers: `priority_memory` (successful episodes) and `memory` (all episodes)
- Dynamic sampling ratio: `0.3 + (success_rate * 0.4)`
  - Minimum 30% from priority memory
  - Maximum 70% from priority memory (when success rate is high)

**Why:**
- Successful episodes contain valuable trajectories
- Sampling them more frequently accelerates learning
- As success rate increases, more priority samples are used

### 2. Epsilon-Greedy Exploration

**Schedule:**
- Initial: `epsilon = 1.0` (100% exploration)
- Decay: `epsilon *= 0.9995` per training step
- Minimum: `epsilon = 0.001` (0.1% exploration)

**Why:**
- Starts with full exploration to discover good strategies
- Gradually transitions to exploitation as learning progresses
- Maintains small exploration to adapt to changes

### 3. Target Network (DQN Standard)

**Update Method:** Soft update (Polyak averaging)
```python
target_param = TAU * policy_param + (1 - TAU) * target_param
```

**Why:**
- Provides stable Q-value targets during learning
- Prevents correlation between current and target Q-values
- Faster updates (TAU=0.01) help with stability

### 4. Gradient Clipping

**Implementation:**
```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
```

**Why:**
- Prevents gradient explosion
- Critical for stability when learning rate is low
- max_norm=0.5 is conservative, ensuring very stable updates

### 5. Q-Value Clipping

**Implementation:**
```python
next_q = torch.clamp(next_q, -200, 200)
```

**Why:**
- Directly prevents Q-value explosion
- Keeps Q-values in reasonable range
- Prevents Q-scores from growing > 500

### 6. Best Model Auto-Save

**Implementation:**
- Automatically saves model when episode reward exceeds `best_score`
- Saved to `best_model.pth`

**Why:**
- Preserves best performing model
- Allows recovery if training degrades
- Enables loading best model for evaluation

---

## Key Features

### 1. Multiple Target Support
- Car can navigate to multiple targets in sequence
- Each target has different color for visualization
- Episode continues until all targets reached

### 2. Real-time Metrics Display
- **Epsilon:** Current exploration rate
- **Last Reward:** Most recent episode reward
- **Q-Score:** Maximum Q-value from current state
- **Loss:** Current training loss
- **Success:** Success count and percentage
- **Overall Time:** Total training time
- **Overall Steps:** Total steps taken
- **Best Score:** Best episode score achieved

### 3. Interactive Map Setup
- Click on map to set car starting position
- Click multiple times to set target sequence
- Right-click to finish setup

### 4. Model Management
- **Save Best Weights:** Manually save current best model
- **Load Weights:** Load previously saved model
- **Auto-save:** Automatically saves when best score improves

### 5. Reward History Chart
- Visualizes last 50 episode scores
- Shows raw scores and 10-episode moving average
- Helps monitor training progress

---

## Usage

### Running the Application

```bash
python citymap_assignment.py
```

### Setup Process

1. **Load Map:**
   - Click "📂 LOAD MAP" to load a city map image
   - Default: `city_map.png` or `city_map.jpg`

2. **Set Car Position:**
   - Click on the map where you want the car to start

3. **Set Targets:**
   - Click multiple times on the map to set target sequence
   - Each target gets a different color
   - Right-click when done

4. **Start Training:**
   - Press SPACE or click "▶ START"
   - Training begins automatically

### Controls

- **SPACE:** Start/Pause training
- **▶ START:** Begin training
- **⏸ PAUSE:** Pause training
- **↺ RESET ALL:** Reset everything
- **💾 SAVE BEST WEIGHTS:** Save current best model
- **📥 LOAD WEIGHTS:** Load saved model

---

## Monitoring Training

### Key Metrics to Watch

#### 1. Success Rate
- **Target:** > 80% for good performance
- **Good:** 50-80% and increasing
- **Poor:** < 20% and not improving

#### 2. Loss
- **Target:** < 1.0 for stable learning
- **Good:** 0.1-1.0 and decreasing
- **Poor:** > 10.0 or increasing

#### 3. Q-Score
- **Target:** 50-200 for this task
- **Good:** 100-300 and stable
- **Poor:** > 500 (indicates Q-value explosion)

#### 4. Best Score
- **Target:** Positive and increasing
- **Good:** > 50 and improving
- **Poor:** Negative or stuck

### When to Stop Training

**Stop when:**
- ✅ Success rate ≥ 80% for 50+ consecutive episodes
- ✅ Loss < 1.0 and stable
- ✅ Q-Score < 500 and stable
- ✅ Best score plateaued at good level

**Continue if:**
- ⏳ Success rate is improving (even if < 50%)
- ⏳ Loss is decreasing
- ⏳ Best score is increasing

**Stop and fix if:**
- ⚠️ Loss > 1000 and not decreasing
- ⚠️ Success rate < 5% for 500+ episodes
- ⚠️ Q-Score > 1000 (Q-value explosion)

---

## Troubleshooting

### Problem: High Loss (> 1000)

**Symptoms:**
- Loss value very high
- Q-Score very high (> 500)
- Training unstable

**Solutions:**
1. Reduce learning rate: `LR = 0.00001`
2. Tighter gradient clipping: `max_norm = 0.1`
3. Smaller batch size: `BATCH_SIZE = 64`
4. Increase TAU: `TAU = 0.05`

### Problem: Low Success Rate (< 5%)

**Symptoms:**
- Very few successful episodes
- Car crashes frequently
- Not learning to navigate

**Solutions:**
1. Increase crash penalties: `-300, -150, -75`
2. Increase target reward: `reward = 200`
3. Reduce step penalty: `reward = -0.01`
4. Increase exploration: `EPSILON_MIN = 0.05`

### Problem: Q-Score Too High (> 500)

**Symptoms:**
- Q-Score growing unbounded
- Q-values exploding

**Solutions:**
1. Clip Q-values: `torch.clamp(next_q, -200, 200)`
2. Reduce gamma: `GAMMA = 0.9`
3. Clip rewards: `torch.clamp(r, -10, 10)`

### Problem: Inconsistent Performance

**Symptoms:**
- Sometimes succeeds, sometimes fails
- High variance in performance

**Solutions:**
1. Faster epsilon decay: `epsilon *= 0.9998`
2. Lower epsilon minimum: `EPSILON_MIN = 0.005`
3. More stable learning: `LR = 0.00005`
4. Load best model and continue training

---

## Hyperparameter Tuning Guide

### For Narrow Roads
- `SENSOR_DIST = 15-20` (short sensors)
- `SPEED = 0.8-1.0` (slower for control)

### For Wide Roads
- `SENSOR_DIST = 80-100` (longer sensors)
- `SPEED = 1.5-2.0` (faster movement)

### For Complex Maps
- `memory = 50000` (more diverse experiences)
- `priority_memory = 20000` (more successful episodes)

### For Faster Learning
- `LR = 0.0001` (higher learning rate, but monitor for instability)
- `TAU = 0.01` (faster target updates)
- `BATCH_SIZE = 256` (larger batches)

### For More Stable Learning
- `LR = 0.00001` (very conservative)
- `max_norm = 0.1` (tighter gradient clipping)
- `BATCH_SIZE = 64` (smaller batches)

---

## Technical Details

### State Normalization
- Distance normalized by: `max(map_width, map_height) * 1.2`
- Angle normalized by: `180.0` degrees
- Sensor values: `brightness / 255.0` (0.0 to 1.0)

### Action Space
- 5 discrete actions
- Actions modify car angle, then move forward
- Turn speeds: 1.0° (regular) or 18° (sharp)

### Episode Termination
- Episode ends when:
  - Car crashes (hits boundary)
  - All targets reached
  - `MAX_CONSECUTIVE_CRASHES` exceeded

### Memory Management
- Experiences stored as: `(state, action, reward, next_state, done)`
- Successful episodes (reward > 0) go to priority memory
- Failed episodes go to regular memory
- Both buffers use FIFO (oldest experiences removed when full)

---

## References

- **DQN Paper:** Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015)
- **Prioritized Experience Replay:** Schaul et al., "Prioritized Experience Replay" (ICLR, 2016)
- **Target Networks:** Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (AAAI, 2016)

---

## License

This is an educational assignment project.

---

## Author Notes

This implementation focuses on:
1. **Stability:** Low learning rates, gradient clipping, Q-value clipping
2. **Efficiency:** Prioritized replay, target networks
3. **Robustness:** Progressive rewards, multiple target support
4. **Monitoring:** Comprehensive metrics display

The hyperparameters were tuned specifically for:
- 1024x1024 city map
- Narrow roads with variable widths
- Multiple target navigation
- Stable learning without Q-value explosion

---

## Version History

- **v1.0:** Initial implementation with optimized hyperparameters
- Added gradient clipping for stability
- Added Q-value clipping to prevent explosion
- Implemented progressive reward structure
- Added comprehensive metrics display

