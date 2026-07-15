import numpy as np
import matplotlib.pyplot as plt
import os

from vehicle_model import Vehicle, LeadCarProfile
from pid_controller import PIDController, ACCSpacingPolicy, SafetyOverride

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_scenario(name, lead_profile_fn, kp, ki, kd, t_end=40.0, dt=0.02,
                  ego_v0=20.0, gap0=35.0):
    ego = Vehicle(position=0.0, velocity=ego_v0)
    lead = Vehicle(position=gap0, velocity=lead_profile_fn(0.0))
    pid = PIDController(kp, ki, kd)
    policy = ACCSpacingPolicy(standstill_gap=5.0, time_gap=1.5)
    safety = SafetyOverride(min_gap=4.0, a_emergency=-6.0, buffer_time=1.1)

    steps = int(t_end / dt)
    t_hist, gap_hist, gap_des_hist, v_ego_hist, v_lead_hist, a_hist = ([] for _ in range(6))
    override_count = 0
    min_gap = float("inf")

    for i in range(steps):
        t = i * dt
        v_lead_target = lead_profile_fn(t)
        # drive lead car's own velocity toward its profile target (simple tracking)
        lead_a = np.clip((v_lead_target - lead.v) * 2.0, -4.0, 2.5)
        lead.step(lead_a, dt)

        gap = lead.x - ego.x
        min_gap = min(min_gap, gap)
        gap_des = policy.desired_gap(ego.v)
        error = gap - gap_des

        a_cmd = pid.update(error, dt)
        closing_speed = ego.v - lead.v
        a_cmd, overridden = safety.apply(gap, closing_speed, a_cmd)
        if overridden:
            override_count += 1
        ego.step(a_cmd, dt)

        t_hist.append(t)
        gap_hist.append(gap)
        gap_des_hist.append(gap_des)
        v_ego_hist.append(ego.v)
        v_lead_hist.append(lead.v)
        a_hist.append(ego.a)

    metrics = compute_metrics(t_hist, gap_hist, gap_des_hist, a_hist)
    metrics["min_gap_m"] = round(min_gap, 3)
    metrics["safety_override_active_pct"] = round(100 * override_count / steps, 2)
    plot_scenario(name, t_hist, gap_hist, gap_des_hist, v_ego_hist, v_lead_hist, a_hist)
    return metrics


def compute_metrics(t, gap, gap_des, a):
    gap = np.array(gap)
    gap_des = np.array(gap_des)
    a = np.array(a)
    err = gap - gap_des

    steady_state_err = np.mean(np.abs(err[-int(len(err) * 0.1):]))  # last 10% of run
    max_overshoot = np.max(-err) if np.min(err) < 0 else 0.0        # gap undershoot = danger
    max_decel = np.min(a)
    rms_jerk = np.sqrt(np.mean(np.diff(a) ** 2)) if len(a) > 1 else 0.0

    return {
        "steady_state_gap_error_m": round(float(steady_state_err), 3),
        "max_gap_undershoot_m": round(float(max_overshoot), 3),
        "max_deceleration_mps2": round(float(max_decel), 3),
        "rms_jerk_mps2_per_step": round(float(rms_jerk), 4),
    }


def plot_scenario(name, t, gap, gap_des, v_ego, v_lead, a):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(t, gap, label="actual gap")
    axs[0].plot(t, gap_des, "--", label="desired gap")
    axs[0].set_ylabel("Gap (m)")
    axs[0].legend()
    axs[0].set_title(f"ACC PID Response — {name}")

    axs[1].plot(t, v_ego, label="ego velocity")
    axs[1].plot(t, v_lead, "--", label="lead velocity")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()

    axs[2].plot(t, a, color="tab:red")
    axs[2].set_ylabel("Ego accel (m/s^2)")
    axs[2].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    # PID gains — tuned by hand via Ziegler-Nichols-style iteration on the step scenario,
    # then verified against the harder decel/accel and stop-and-go scenarios.
    KP, KI, KD = 0.55, 0.09, 0.22

    scenarios = {
        "step_change": LeadCarProfile.step_change,
        "decel_then_accel": LeadCarProfile.decel_then_accel,
        "stop_and_go": LeadCarProfile.stop_and_go,
    }

    scenario_overrides = {
        # aggressive braking-cycle scenario needs a larger safety cushion and lower
        # entry speed to stay physically achievable under the -4 m/s^2 decel limit
        "stop_and_go": {"ego_v0": 14.0, "gap0": 40.0},
    }

    print(f"PID gains: Kp={KP} Ki={KI} Kd={KD}\n")
    for name, profile_fn in scenarios.items():
        kwargs = scenario_overrides.get(name, {})
        metrics = run_scenario(name, profile_fn, KP, KI, KD, **kwargs)
        print(f"[{name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print()
