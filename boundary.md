Checking the reward structure and how boundary signals are handled:


[4 tools called]


## What is "boundary-signal"?

In your code, "boundary-signal" can refer to:

1. Crash detection signal: `car_center_val < 0.4` (line 281) — detects when car hits boundary/off-road
2. Sensor boundary detection: sensor brightness values that detect road vs off-road ahead
3. Crash reward: `-100` penalty when hitting boundary

---

## Current reward structure

```python
Crash (boundary hit):     -100
Target reached:           +100
Step penalty:             -0.05
Orientation bonus:        up to +20
Distance penalty:          -8
```

---

## What happens when boundary-signal is weak

### Scenario 1: Weak crash penalty (relative to rewards)

If crash penalty is too small compared to positive rewards:

Problem:
- Agent may risk crashes to reach targets faster
- Short-term gains outweigh crash penalties
- Agent learns risky behavior

Example:
```python
# Weak boundary signal:
Crash: -50  (too small)
Target: +100
# Agent thinks: "Crash is only -50, target is +100, worth the risk!"
```

Current situation:
```python
Crash: -100
Target: +100
# Ratio: 1:1 (balanced, but could be stronger)
```

---

### Scenario 2: Weak sensor detection (low brightness values)

If sensor values are weak (low brightness = weak boundary signal):

Problem:
- Sensors don't clearly detect boundaries ahead
- Agent can't see obstacles coming
- More crashes because of poor detection

Example:
```python
# Weak sensor signal:
sensor_val = 0.3  # Weak signal (can't tell if road or boundary)
# Agent can't distinguish road from boundary
```

Current situation:
```python
# Sensors return brightness/255.0
# Road (white): ~1.0
# Boundary (dark): ~0.0-0.3
# Crash threshold: < 0.4
```

---

### Scenario 3: Weak crash threshold (too lenient)

If `car_center_val < 0.4` is too lenient:

Problem:
- Car can go partially off-road without crashing
- Agent learns it's okay to be near boundaries
- More boundary-hugging behavior

Current situation:
```python
if car_center_val < 0.4:  # Crash threshold
    reward = -100
```

---

## Effects of weak boundary-signal

### 1. Risk-taking behavior

```
Strong boundary signal:  Crash = -200, Target = +100
→ Agent avoids boundaries (risk-averse)

Weak boundary signal:     Crash = -50, Target = +100  
→ Agent takes risks (risk-seeking)
```

### 2. Poor learning

- Agent doesn't learn to avoid boundaries
- Q-values don't properly penalize boundary states
- Training becomes unstable

### 3. Inconsistent performance

- Sometimes succeeds (lucky)
- Sometimes crashes (risky behavior)
- High variance in performance

---

## Solutions for weak boundary-signal

### Solution 1: Increase crash penalty

```python
# Current:
if car_center_val < 0.4:
    reward = -100

# Stronger boundary signal:
if car_center_val < 0.4:
    reward = -200  # or -300 (2-3x target reward)
```

Why:
- Makes crashes more costly
- Agent learns to avoid boundaries
- More risk-averse behavior

---

### Solution 2: Stricter crash threshold

```python
# Current:
if car_center_val < 0.4:  # Too lenient

# Stricter:
if car_center_val < 0.5:  # or 0.6 (stricter boundary)
    reward = -100
```

Why:
- Detects boundary earlier
- Prevents boundary-hugging
- Clearer signal

---

### Solution 3: Progressive boundary penalty

```python
# Gradual penalty as approaching boundary:
if car_center_val < 0.4:
    reward = -100  # Full crash
elif car_center_val < 0.5:
    reward = -20   # Warning penalty (near boundary)
elif car_center_val < 0.6:
    reward = -5    # Small warning
```

Why:
- Teaches agent to avoid boundaries gradually
- Better learning signal
- Smoother behavior

---

### Solution 4: Increase sensor sensitivity

```python
# Current sensor detection:
val = brightness / 255.0

# More sensitive (if needed):
val = (brightness / 255.0) ** 0.8  # Non-linear scaling
# or
val = brightness / 200.0  # More sensitive
```

Why:
- Better boundary detection
- Clearer signals to network
- Better state representation

---

## Recommended fix for your case

Given you have 6 successes but inconsistent performance, strengthen the boundary signal:

```python
# Option 1: Stronger crash penalty (RECOMMENDED)
if car_center_val < 0.4:
    reward = -200  # Increased from -100 (2x target reward)

# Option 2: Stricter threshold + stronger penalty
if car_center_val < 0.5:  # Stricter
    reward = -150  # Stronger penalty
```

---

## Why this helps consistency

Current problem:
```
Crash: -100, Target: +100
→ Agent sometimes risks crashes for targets
→ Inconsistent: sometimes succeeds, sometimes crashes
```

With stronger boundary signal:
```
Crash: -200, Target: +100
→ Agent avoids boundaries more carefully
→ More consistent: fewer crashes, more reliable paths
```

---

## Summary

When boundary-signal is weak compared to rewards:

1. Agent takes more risks
2. Learning is unstable
3. Performance is inconsistent
4. Q-values don't properly penalize boundaries

Solution: strengthen the boundary signal by:
- Increasing crash penalty (e.g., -200 instead of -100)
- Stricter crash threshold (e.g., 0.5 instead of 0.4)
- Progressive penalties near boundaries

For your case, try increasing crash penalty to `-200` or `-300` to make boundary avoidance more important than reaching targets quickly.