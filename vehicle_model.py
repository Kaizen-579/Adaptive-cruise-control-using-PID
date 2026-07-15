"""
Longitudinal vehicle dynamics for a leader-follower (ego + lead car) pair.

Model: first-order lag between commanded acceleration and actual acceleration
(represents powertrain/brake actuator delay), plus simple drag.
"""

import numpy as np


class Vehicle:
    def __init__(self, position=0.0, velocity=0.0, tau=0.5, drag_coeff=0.02):
        """
        position   : m
        velocity   : m/s
        tau        : actuator lag time constant (s) - how fast accel commands
                     actually translate to real acceleration (throttle/brake dynamics)
        drag_coeff : simple velocity-proportional drag (1/s), models air resistance
        """
        self.x = position
        self.v = velocity
        self.a = 0.0
        self.tau = tau
        self.drag_coeff = drag_coeff

    def step(self, a_cmd, dt):
        """Advance vehicle state by dt given a commanded acceleration."""
        # first-order actuator lag
        self.a += (a_cmd - self.a) * (dt / self.tau)
        # drag reduces effective acceleration slightly at speed
        a_eff = self.a - self.drag_coeff * self.v
        self.v += a_eff * dt
        self.v = max(self.v, 0.0)  # no reverse rolling
        self.x += self.v * dt
        return self.x, self.v, self.a


class LeadCarProfile:
    """Generates a lead-vehicle velocity trace for test scenarios."""

    @staticmethod
    def step_change(t, v0=20.0, v1=15.0, t_change=10.0):
        return v1 if t >= t_change else v0

    @staticmethod
    def decel_then_accel(t, v0=25.0, v_min=12.0, t1=8.0, t2=16.0, t3=24.0):
        if t < t1:
            return v0
        elif t < t2:
            frac = (t - t1) / (t2 - t1)
            return v0 + frac * (v_min - v0)
        elif t < t3:
            frac = (t - t2) / (t3 - t2)
            return v_min + frac * (v0 - v_min)
        else:
            return v0

    @staticmethod
    def stop_and_go(t):
        cycle = t % 24
        return 0.0 if cycle < 6 else 14.0
