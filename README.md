# Adaptive-cruise-control-using-PID

A Python simulation of longitudinal Adaptive Cruise Control using a PID
spacing controller with a constant time-gap policy, plus a hard safety
override layer. Includes a leader-follower vehicle dynamics model with
first-order actuator lag and drag.

## Why this exists

Demonstrates a full control-design loop: plant modeling → controller design →
tuning → closed-loop evaluation against quantitative metrics → identifying and
fixing a real failure mode, not just "it looks stable on a plot."

## Architecture

```
src/
  vehicle_model.py    # Vehicle (1st-order actuator lag + drag) and lead-car velocity profiles
  pid_controller.py   # PIDController (anti-windup), ACCSpacingPolicy, SafetyOverride
  simulate.py          # Runs scenarios, computes metrics, saves plots to results/
```

## Control design

**Spacing policy** — constant time-gap (standard in production ACC):

    desired_gap = standstill_gap + time_gap * v_ego

**PID loop** — `error = actual_gap - desired_gap`, output is commanded ego
acceleration, saturated to `[-4.0, 2.5] m/s²`. Anti-windup clamps the integral
accumulator to `a_limits / Ki` so integral action can't be silently capped
below what the actuator can actually deliver (an earlier fixed-bound version
of this had exactly that bug — integral contribution was pinned to ~0.27 m/s²
regardless of how large the gap error got).

**Safety override** — a linear PID reacting only to gap error cannot be tuned
to guarantee collision avoidance against a lead vehicle braking hard and
repeatedly (verified experimentally below). Every production ACC pairs the
comfort-oriented spacing loop with a separate safety-critical minimum-distance
check. This repo includes a minimal version: a closing-rate-aware trigger
(`min_gap + closing_speed * buffer_time`) that overrides the PID with max
braking when violated — a simplified time-to-collision (TTC) guard rather than
a fixed distance threshold, since a fixed threshold reacts too late at high
closing speed.

## Scenarios evaluated

| Scenario | Description |
|---|---|
| `step_change` | Lead vehicle drops speed once (20→15 m/s) |
| `decel_then_accel` | Lead vehicle ramps down then back up (25→12→25 m/s) |
| `stop_and_go` | Repeated hard braking to a stop and back up — the stress test |

## Results (Kp=0.55, Ki=0.09, Kd=0.22)

| Scenario | Steady-state gap error | Max deceleration | Min gap | Safety override active |
|---|---|---|---|---|
| step_change | 0.02 m | -1.81 m/s² | 25.9 m | 0% |
| decel_then_accel | 0.17 m | -1.54 m/s² | 24.0 m | 0% |
| stop_and_go | 7.49 m | -6.00 m/s² | 1.69 m | 19.2% |

**Honest finding:** in the `stop_and_go` stress scenario, no PID gain
combination I tried (including much more aggressive gains) prevented the gap
from going negative (a simulated collision) without the safety override layer.
This is a structural limitation of reactive linear control against repeated
hard braking, not a tuning failure — it's the reason real ACC systems are not
"just a PID." Plots for all three scenarios are in `results/`.

## Run it

```bash
pip install numpy matplotlib
python3 src/simulate.py
```

## Possible extensions

- Replace PID with MPC to get predictive (not just reactive) braking margin
- Add sensor noise + a Kalman filter for gap/relative-velocity estimation
- Move from kinematic sim to CARLA/Simulink for closer-to-hardware validation
- Port the control loop to an ESP32 + ultrasonic/LiDAR rig for a physical demo
