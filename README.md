# NeuralNav: TD3-Based Autonomous Car Navigation

A PyQt6-based reinforcement learning application that uses **Twin Delayed Deep Deterministic Policy Gradient (TD3)** to train an autonomous car to navigate through a city map and reach multiple targets in sequence.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Hyperparameters](#hyperparameters)
- [Training Strategies](#training-strategies)
- [Reward Function](#reward-function)
- [Learning Metrics](#learning-metrics)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a continuous control reinforcement learning system where a car learns to:
- Navigate through a city map using sensor inputs
- Avoid obstacles (off-road areas)
- Reach multiple targets in sequence
- Learn from both successful and failed episodes

**Key Features:**
- **TD3 Algorithm**: State-of-the-art actor-critic method for continuous control
- **Continuous Action Space**: 2D actions [steering, throttle]
- **Multiple Targets**: Sequential target navigation
- **Adaptive Prioritized Experience Replay**: Learns more from successful episodes
- **Repeat Mistake Detection**: Prevents getting stuck in loops
- **Comprehensive Learning Metrics**: Real-time monitoring of training progress

---

## Architecture

### TD3 (Twin Delayed DDPG)

TD3 is an improvement over DDPG that addresses overestimation bias in Q-learning through:

1. **Twin Critic Networks**: Two Q-networks that take the minimum to reduce overestimation
2. **Target Policy Smoothing**: Adds noise to target actions for regularization
3. **Delayed Policy Updates**: Actor updates less frequently than critic (every 2 steps)

### Network Architecture

#### Actor Network
- **Input**: 9-dimensional state vector
  - 7 sensor readings (at angles: -45°, -30°, -15°, 0°, 15°, 30°, 45°)
  - Normalized angle to target
  - Normalized distance to target
- **Output**: 2D continuous action [steering, throttle]
- **Architecture**: 
  ```
  Input(9) → Linear(256) → LayerNorm → ReLU
         → Linear(512) → LayerNorm → ReLU
         → Linear(512) → LayerNorm → ReLU
         → Linear(256) → LayerNorm → ReLU
         → Linear(2) → Tanh → Scaled Output
  ```
- **Normalization**: LayerNorm (instead of BatchNorm) to handle batch size 1 during inference
- **Initialization**: Throttle bias set to 0.5 to encourage initial movement

#### Critic Network (Twin)
- **Input**: State (9D) + Action (2D) = 11D
- **Output**: Q-value (single scalar)
- **Architecture**: Two identical networks (Q1 and Q2) with same structure as Actor
- **Purpose**: Take minimum of Q1 and Q2 to reduce overestimation

---

## Hyperparameters

### Core TD3 Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **BATCH_SIZE** | 512 | Larger batches provide more stable gradients. 512 balances stability with update frequency. |
| **GAMMA** | 0.98 | Discount factor. 0.98 emphasizes long-term rewards, important for multi-target navigation. |
| **LR_ACTOR** | 0.0005 | Lower learning rate for actor prevents policy from changing too rapidly, ensuring stable learning. |
| **LR_CRITIC** | 0.001 | Higher learning rate for critic allows faster Q-value convergence. |
| **TAU** | 0.01 | Polyak averaging coefficient. 0.01 (vs standard 0.005) provides faster target network updates while maintaining stability. |
| **POLICY_NOISE** | 0.2 | Standard deviation of noise added to target policy. Prevents overfitting to Q-function. |
| **NOISE_CLIP** | 0.5 | Clips target policy noise to [-0.5, 0.5]. Prevents extreme actions during training. |
| **POLICY_FREQ** | 2 | Actor updates every 2 critic updates. Standard TD3 setting that prevents policy from changing too fast. |

### Exploration Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **EXPL_NOISE** | 0.2 | Initial exploration noise. Higher value (0.2 vs 0.1) ensures better exploration early in training. |
| **Exploration Decay** | 0.9998 (high) / 0.9999 (low) | Slower decay maintains exploration longer, preventing premature convergence to suboptimal policies. |

### Action Space Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **MAX_STEERING** | 20.0° | Reduced from 25° to prevent tight circles and encourage smoother navigation. |
| **MAX_THROTTLE** | 2.0 | Maximum speed multiplier. Allows 2x base speed for faster learning. |
| **MIN_THROTTLE** | 0.5 | Minimum throttle ensures car always moves forward, preventing getting stuck. |

### Physics Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **SPEED** | 1 | Base speed in pixels per step. Lower value (1 vs 2) provides finer control. |
| **SENSOR_DIST** | 16 | Sensor range in pixels. Short range forces car to react quickly to obstacles. |
| **SENSOR_ANGLE** | 45° | Maximum sensor angle. 7 sensors cover -45° to +45° range. |

### Training Control

| Parameter | Value | Reason |
|-----------|-------|--------|
| **MAX_CONSECUTIVE_CRASHES** | 2 | After 2 consecutive crashes, reset to origin. Prevents wasting time in impossible situations. |
| **Warm-up Steps** | 1000 | Collects experiences before training starts. Ensures sufficient data for stable learning. |

---

## Training Strategies

### 1. Adaptive Prioritized Experience Replay

**Strategy**: Separate memory buffers for high-reward and regular episodes.

**Implementation**:
- `priority_memory`: Stores experiences from successful episodes (reward > 0)
- `regular_memory`: Stores all other experiences
- **Adaptive Sampling Ratio**: 30-70% priority samples based on success rate
  - Early training: ~30% priority (few successes available)
  - Later training: ~60-70% priority (many successes available)

**Reasoning**:
- Successful episodes contain valuable information about good policies
- Learning from successes accelerates convergence
- Adaptive ratio ensures we don't ignore failures completely

### 2. Repeat Mistake Detection

**Strategy**: Track recent state-action pairs and penalize repetition.

**Implementation**:
- Maintains last 50 states and actions
- Detects similar state-action pairs (within thresholds)
- Applies penalty: `-2.0 * repeat_count` for repeated patterns

**Reasoning**:
- Prevents car from getting stuck in loops
- Forces exploration of different strategies
- Breaks repetitive behavior patterns

### 3. Progress-Based Reward Shaping

**Strategy**: Strong emphasis on forward progress toward target.

**Implementation**:
- Progress reward: `8.0 * (distance_reduction)`
- Moving away penalty: `-30`
- Significant progress bonus: `+3.0` if progress > 3 pixels

**Reasoning**:
- Progress is the most important signal for navigation
- Strong penalties for moving away prevent circling
- Bonuses encourage efficient pathfinding

### 4. Directional Guidance

**Strategy**: Reward pointing toward target.

**Implementation**:
- Direction reward: `3.0 * (1 - |angle_to_target|)`
- Maximum reward when directly pointing at target

**Reasoning**:
- Helps car orient correctly toward target
- Reduces unnecessary turning
- Encourages efficient navigation

### 5. Smooth Driving Incentives

**Strategy**: Penalize excessive steering and reward balanced sensor readings.

**Implementation**:
- Steering penalty: `-1.0 * (|steering| / MAX_STEERING)`
- Sensor balance reward: `+1.5 * min(left_sensors, right_sensors)`

**Reasoning**:
- Encourages smooth, controlled driving
- Prevents erratic steering behavior
- Rewards staying centered on road

### 6. Stuck Behavior Detection

**Strategy**: Track progress over multiple steps and penalize stagnation.

**Implementation**:
- Maintains progress history over last 30 steps
- If average progress < -0.5: penalty of `-10.0`

**Reasoning**:
- Detects when car is circling or stuck
- Provides strong signal to change behavior
- Prevents wasting time in unproductive states

### 7. Slower Exploration Decay

**Strategy**: Maintain exploration longer to avoid local minima.

**Implementation**:
- Decay rate: 0.9998 (high noise) → 0.9999 (low noise)
- Slower than standard 0.9995

**Reasoning**:
- Prevents premature convergence to suboptimal policies
- Maintains ability to explore new strategies
- Helps escape local minima

---

## Reward Function

The reward function is carefully designed to guide learning:

### Terminal Rewards
- **Crash**: `-100` (strong negative signal)
- **Target Reached**: `+100` (strong positive signal)

### Step Rewards (Normal Operation)
1. **Base Step Penalty**: `-0.1` (encourages efficiency)
2. **Road Detection**: `+3.0 * center_sensor` (staying on road)
3. **Progress Reward**: `+8.0 * progress` (most important - forward movement)
4. **Moving Away Penalty**: `-30` (strong discouragement)
5. **Significant Progress Bonus**: `+3.0` (if progress > 3 pixels)
6. **Steering Penalty**: `-1.0 * (|steering| / MAX_STEERING)` (smooth driving)
7. **Direction Reward**: `+3.0 * (1 - |angle_to_target|)` (pointing at target)
8. **Speed Reward**: `+0.3 * throttle` (maintaining movement)
9. **Repeat Penalty**: `-2.0 * repeat_count` (avoiding loops)
10. **Stuck Penalty**: `-10.0` (if average progress < -0.5)
11. **Sensor Balance**: `+1.5 * min(left, right)` (centered on road)

**Total Reward Range**: Approximately `-150` (crash) to `+150` (successful navigation)

---

## Learning Metrics

The UI displays comprehensive metrics to monitor training:

### Primary Indicators

1. **Q-Score**: Average Q-value (can be negative - normal with negative rewards)
   - **Trend Indicator**: ↑ (improving), → (stable), ↓ (declining)
   - **Color**: Green if > -50, Red if worse

2. **Average Episode Reward**: Most important metric
   - Should **increase** over time if learning
   - Positive values indicate good performance

3. **Average Episode Length**: Should **increase** (car survives longer)

4. **Crash Rate**: Should **decrease** (fewer crashes)

5. **Average Distance to Target**: Should **decrease** (getting closer)

6. **Successes**: Number of targets reached (should **increase**)

### How to Know if Learning is Working

✅ **Signs of Learning:**
- Q Trend shows ↑ (improving)
- Avg Reward increasing
- Avg Length increasing
- Crash Rate decreasing
- Avg Distance decreasing
- Successes increasing

❌ **Signs of Not Learning:**
- Q Trend shows ↓ or → (stuck)
- Avg Reward flat or decreasing
- Crash Rate staying high (>80%)
- Same mistakes repeating

---

## Usage

### Setup

1. **Install Dependencies**:
   ```bash
   pip install torch PyQt6 numpy
   ```

2. **Run the Application**:
   ```bash
   python3 citymap.py
   ```

### Basic Workflow

1. **Load Map**: Click "📂 LOAD MAP" or use default `city_map.png`
2. **Set Car Position**: Click on map where car should start
3. **Set Targets**: Click on map to add targets (can add multiple)
   - Right-click when done adding targets
4. **Start Training**: Press SPACE or click "▶ START"
5. **Monitor Progress**: Watch the metrics panel and reward chart

### Controls

- **SPACE**: Start/Pause training
- **↺ RESET ALL**: Reset everything (car, targets, training)
- **💾 SAVE BEST WEIGHTS**: Save current best model
- **📂 LOAD WEIGHTS**: Load previously saved model

### Tips for Better Training

1. **Start Simple**: Use a simple map with clear paths
2. **Place Targets Strategically**: Start with 1-2 targets, add more as performance improves
3. **Monitor Metrics**: Watch Q Trend and Avg Reward
4. **Adjust Hyperparameters**: If not learning, try:
   - Increase `EXPL_NOISE` (more exploration)
   - Decrease `LR_ACTOR` (more stable)
   - Increase `BATCH_SIZE` (more stable gradients)
5. **Save Weights**: Save best weights when performance is good

---

## Troubleshooting

### Car Not Moving
- **Check**: Speed display should show > 0
- **Fix**: Ensure `MIN_THROTTLE` is high enough (≥ 0.5)
- **Fix**: Check that actions are being clipped correctly

### Car Circling/Repeating Mistakes
- **Check**: Q Trend should show ↑
- **Fix**: Increase `EXPL_NOISE` to encourage exploration
- **Fix**: Check reward function - progress rewards should be strong
- **Fix**: Reduce `MAX_STEERING` to prevent tight turns

### Q-Score Not Improving
- **Remember**: Negative Q-scores are normal (rewards are mostly negative)
- **Check**: Q Trend indicator (↑ = improving, even if negative)
- **Check**: Avg Reward should be increasing
- **Fix**: If Q Trend is ↓, try:
  - Lower learning rates
  - Increase batch size
  - Check reward scaling

### Training Too Slow
- **Fix**: Reduce `BATCH_SIZE` to 256 (faster updates)
- **Fix**: Increase `TAU` to 0.02 (faster target updates)
- **Fix**: Reduce network size if needed

### Training Unstable
- **Fix**: Increase `BATCH_SIZE` to 512 or higher
- **Fix**: Lower learning rates
- **Fix**: Increase `TAU` slightly (0.01 → 0.005)

### Not Reaching Targets
- **Check**: Reward function - progress rewards should be high
- **Fix**: Increase `EXPL_NOISE` to explore more
- **Fix**: Simplify map or reduce number of targets initially
- **Fix**: Check if targets are reachable (not in obstacles)

---

## File Structure

```
assignment/
├── citymap.py          # Main application file
├── city_map.png        # Default map image
├── best_weights.pth    # Auto-saved best weights
└── README.md           # This file
```

---

## Key Design Decisions

### Why TD3 over DDPG?
- **Twin Critics**: Reduces overestimation bias in Q-values
- **Target Smoothing**: Prevents overfitting to Q-function
- **Delayed Updates**: More stable policy learning

### Why Continuous Actions?
- **Smoother Control**: Better than discrete actions for navigation
- **More Efficient**: Learns optimal steering angles and speeds
- **Realistic**: Matches real-world control systems

### Why Adaptive Prioritized Replay?
- **Faster Learning**: Focuses on successful experiences
- **Balanced**: Still learns from failures
- **Adaptive**: Adjusts based on success rate

### Why Repeat Mistake Detection?
- **Prevents Loops**: Actively breaks repetitive behavior
- **Forces Exploration**: Encourages trying new strategies
- **Efficient**: Doesn't waste time in unproductive states

### Why LayerNorm over BatchNorm?
- **Batch Size 1**: Works during inference (single action selection)
- **Consistent**: Same behavior during training and inference
- **Stable**: No batch statistics needed

---

## References

- **TD3 Paper**: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)
- **DDPG Paper**: "Continuous Control with Deep Reinforcement Learning" (Lillicrap et al., 2015)

---

## License

This project is for educational purposes as part of the ERA (End-to-End Reinforcement Learning) course.

---

## Author Notes

This implementation focuses on:
- **Stability**: Careful hyperparameter tuning for stable learning
- **Efficiency**: Adaptive strategies to learn faster
- **Robustness**: Multiple mechanisms to prevent common RL pitfalls
- **Observability**: Comprehensive metrics to understand learning progress

The reward function is the most critical component - it shapes what the agent learns. The current design emphasizes forward progress while discouraging circling and repetition.

