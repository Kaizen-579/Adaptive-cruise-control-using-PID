"""
PID controller for Adaptive Cruise Control.

constant time-gap spacing policy.
    desired_gap = standstill_gap + time_gap * v_ego
    error = actual_gap - desired_gap

Output: commanded acceleration for the ego vehicle, saturated to realistic limits.
"""


class PIDController:
    def __init__(self, kp, ki, kd, a_min=-4.0, a_max=2.5, i_min=None, i_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.a_min, self.a_max = a_min, a_max   # acceleration saturation (m/s^2)
        # anti-windup clamp on the integral ACCUMULATOR, sized so that ki * integral
        # alone can never exceed actuator authority (a_min/a_max). Fixed arbitrary
        # bounds here silently cap integral action far below what's needed.
        self.i_min = i_min if i_min is not None else a_min / ki
        self.i_max = i_max if i_max is not None else a_max / ki
        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def update(self, error, dt):
        self._integral += error * dt
        self._integral = max(self.i_min, min(self.i_max, self._integral))  # anti-windup

        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        u_sat = max(self.a_min, min(self.a_max, u))

        # clamp integral further if output saturated (conditional anti-windup)
        if u != u_sat:
            self._integral -= error * dt

        return u_sat


class SafetyOverride:
    """
    Hard minimum-gap override, layered on top of the PID output.

    This exists because a linear PID reacting only to (gap - desired_gap) cannot
    be tuned to guarantee collision avoidance under repeated hard lead-vehicle
    braking - the controller only "knows" about closing rate through the
    derivative term, with no forward-looking margin. Every production ACC system
    pairs the comfort-oriented PID/MPC spacing loop with a separate safety-critical
    minimum-distance (or time-to-collision) override. This is a minimal version
    of that: below `critical_gap`, ignore the PID and command max braking.
    """

    def __init__(self, min_gap=3.0, a_emergency=-6.0, buffer_time=0.8):
        self.min_gap = min_gap            # absolute floor, never allow closer than this
        self.a_emergency = a_emergency    # ADAS-grade emergency braking (~0.6g)
        self.buffer_time = buffer_time    # seconds of reaction/margin baked into the trigger

    def apply(self, gap, closing_speed, a_cmd):
        """closing_speed = v_ego - v_lead (positive = ego approaching lead)."""
        # trigger distance = hard floor + however far we'd close in buffer_time
        # at the current closing rate (a simple TTC-style margin, not just fixed distance)
        trigger_gap = self.min_gap + max(0.0, closing_speed) * self.buffer_time
        if gap < trigger_gap:
            return self.a_emergency, True
        return a_cmd, False


class ACCSpacingPolicy:
    """Constant time-gap spacing policy, standard for production ACC systems."""

    def __init__(self, standstill_gap=5.0, time_gap=1.5):
        self.standstill_gap = standstill_gap
        self.time_gap = time_gap

    def desired_gap(self, v_ego):
        return self.standstill_gap + self.time_gap * v_ego
