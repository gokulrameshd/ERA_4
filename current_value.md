Collecting all parameter values from the code:


[7 tools called]


## All parameter values

### 1. Physics parameters

```python
CAR_WIDTH = 14                    # Car width in pixels
CAR_HEIGHT = 8                     # Car height in pixels
SENSOR_DIST = 50                   # Sensor look-ahead distance (pixels)
SENSOR_ANGLE = 5                   # Angle spread of sensors (degrees) - not directly used
SPEED = 1.5                        # Forward speed (pixels per step)
TURN_SPEED = 0.5                   # Regular turn angle (degrees per step)
SHARP_TURN = 5                     # Sharp turn angle for tight corners (degrees)
```

### 2. Reinforcement learning hyperparameters

```python
BATCH_SIZE = 256                   # Experiences sampled per training step
GAMMA = 0.9                        # Discount factor for future rewards (0 to 1)
LR = 0.001                         # Learning rate for Adam optimizer
TAU = 0.001                        # Polyak averaging coefficient for target network
MAX_CONSECUTIVE_CRASHES = 5        # Reset after this many crashes
```

### 3. Exploration/exploitation

```python
epsilon_initial = 1.00             # Starting exploration rate (100% random)
epsilon_min = 0.001                # Minimum exploration rate
epsilon_decay = 0.99995            # Decay factor per training step
```

### 4. Neural network architecture

```python
input_dim = 9                      # State dimensions (7 sensors + angle + distance)
output_dim = 5                     # Number of actions
network_layers = [128, 256, 256, 128]  # Hidden layer sizes
activation = ReLU                  # Activation function
```

### 5. Memory/replay buffer

```python
memory_size = 10000                 # Regular experience replay buffer size
priority_memory_size = 3000         # Priority buffer for successful episodes
episode_scores_buffer = 100        # Track last 100 episode scores
```

### 6. Sensor configuration

```python
num_sensors = 7                    # Number of sensors
sensor_angles = [-45, -30, -15, 0, 15, 30, 45]  # Sensor angles in degrees
```

### 7. Reward values

```python
reward_crash = -100                # Reward for crashing (off-road)
reward_target = 100                # Reward for reaching target
reward_step = -0.1                 # Time penalty per step
reward_orientation_bonus = 20      # Multiplier for orientation reward
reward_orientation_base = 1.0      # Base value for orientation calculation
reward_distance_penalty = 10       # Penalty for moving away from target
target_reach_distance = 20         # Distance threshold to reach target (pixels)
crash_threshold = 0.4              # Brightness threshold for crash detection (< 0.4 = crash)
```

### 8. State normalization

```python
distance_normalization = 800.0     # Distance normalization factor
angle_normalization = 180.0        # Angle normalization factor (degrees)
```

### 9. Prioritized replay sampling

```python
priority_ratio_min = 0.3            # Minimum priority sampling ratio (30%)
priority_ratio_max = 0.7            # Maximum priority sampling ratio (70%)
priority_ratio_formula = 0.3 + (success_rate * 0.4)  # Dynamic ratio
min_batch_size = BATCH_SIZE // 2   # Minimum batch size to train (128)
```

### 10. Initial positions (defaults)

```python
start_pos_x = 100                  # Default start X position
start_pos_y = 100                  # Default start Y position
target_pos_x = 200                 # Default target X position
target_pos_y = 200                 # Default target Y position
initial_car_angle = random(0-360)  # Random initial angle on reset
```

### 11. Model saving/loading

```python
best_model_path = "best_model.pth"  # Default path for best model auto-save
```

### 12. UI/timing

```python
sim_timer_interval = 16            # Game loop timer interval (milliseconds) ~60 FPS
```

### 13. Action space

```python
actions = {
    0: "Left turn",                # -TURN_SPEED degrees
    1: "Straight",                  # 0 degrees
    2: "Right turn",                # +TURN_SPEED degrees
    3: "Sharp left turn",           # -SHARP_TURN degrees
    4: "Sharp right turn"           # +SHARP_TURN degrees
}
```

### 14. Calculated values

```python
# Reaction time (steps)
reaction_steps = SENSOR_DIST / SPEED  # = 50 / 1.5 = ~33 steps

# Steps for 90° turn
steps_for_90_degree_turn = 90 / TURN_SPEED  # = 90 / 0.5 = 180 steps

# Steps for sharp turn (5°)
steps_for_sharp_turn = 1  # Single step
```

### 15. Epsilon decay timeline

| Steps | Epsilon | Exploration % | Exploitation % |
|-------|----------|----------------|----------------|
| 0 | 1.000 | 100% | 0% |
| 200 | ~0.990 | 99% | 1% |
| 2000 | ~0.905 | 90.5% | 9.5% |
| 4600 | ~0.790 | 79% | 21% |
| 9200 | ~0.625 | 62.5% | 37.5% |
| 18400 | ~0.390 | 39% | 61% |
| 46000 | ~0.100 | 10% | 90% |
| 92000+ | ~0.001 | 0.1% | 99.9% |

---

## Summary table

| Category | Parameter | Value |
|----------|-----------|-------|
| **Speed** | `SPEED` | 1.5 pixels/step |
| **Sensors** | `SENSOR_DIST` | 50 pixels |
| **Turns** | `TURN_SPEED` | 0.5°/step |
| **Sharp Turns** | `SHARP_TURN` | 5° |
| **Learning** | `LR` | 0.001 |
| **Batch** | `BATCH_SIZE` | 256 |
| **Discount** | `GAMMA` | 0.9 |
| **Target Update** | `TAU` | 0.001 |
| **Exploration** | `epsilon` | 1.0 → 0.001 |
| **Crash Limit** | `MAX_CONSECUTIVE_CRASHES` | 5 |

These are the current values in your code.