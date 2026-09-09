"""
Module 5: Mathematical Early Warning Engine & Dynamic Escalation Diagnostics
=============================================================================
Computes mathematical trajectories, escalation velocities, predictive lead-time metrics,
and multi-modal hazard indices in real-time.

Mathematical Formulations:
--------------------------
1. Temporal Violence Probability Trajectory:
   P_traj = [P(1), P(2), ..., P(T)],  where P(t) in [0, 1]

2. Dynamic Escalation Velocity (First Derivative / Slope):
   S(t) = (P(t) - P(max(1, t - k))) / (k * delta_t_frame)

3. Dynamic Escalation Acceleration (Second Derivative / Curvature):
   A(t) = (S(t) - S(t - 1)) / delta_t_frame

4. Predictive Lead-Time Metric (Delta t_lead):
   Delta t_lead = (tau_crit - P(t)) / max(S(t), epsilon)
   Forecasts the estimated seconds remaining before violence escalates beyond tau_crit.

5. Composite Multi-Modal Hazard Index:
   H(t) = w_1 * P(V_{t+Delta t}) + w_2 * max(0, S(t)) + w_3 * c_max^{weapon} + w_4 * a_anomaly
   where w_1=0.45, w_2=0.25, w_3=0.20, w_4=0.10

6. Critical Crossing Step:
   t* = min { t in [1, T] | H(t) >= tau_alert }
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class EscalationTrajectoryAnalyzer:
    """
    Real-Time Mathematical Analyzer for Multi-Modal Escalation and Early Warning.
    """
    def __init__(
        self,
        critical_threshold: float = 0.65,
        alert_threshold: float = 0.47,
        fps: float = 5.33,  # 16 frames spanning ~3.0 seconds -> ~5.33 fps
        k_lookback: int = 3
    ):
        self.tau_crit = critical_threshold
        self.tau_alert = alert_threshold
        self.fps = fps
        self.dt_frame = 1.0 / max(1e-3, fps)
        self.k_lookback = k_lookback

    def compute_trajectory_metrics(
        self,
        probabilities: List[float],
        future_prob: Optional[float] = None,
        weapon_threat: float = 0.0,
        audio_anomaly: float = 0.0
    ) -> Dict[str, Any]:
        """
        Args:
            probabilities: List of progressive violence probabilities [P(1), ..., P(T)]
            future_prob: P(V_{t+Delta t}) prediction from early warning model
            weapon_threat: Maximum weapon confidence or composite threat score in [0, 1]
            audio_anomaly: Audio anomaly energy/confidence score in [0, 1]
        Returns:
            Comprehensive diagnostic dictionary of escalation indicators.
        """
        probs = np.array(probabilities, dtype=np.float32)
        T = len(probs)
        if T == 0:
            return {
                "trajectory": [],
                "escalation_slope": 0.0,
                "escalation_acceleration": 0.0,
                "lead_time_seconds": None,
                "hazard_index": 0.0,
                "crossing_step": None,
                "pattern": "NO_DATA",
                "is_escalating": False,
                "early_warning_alert": False
            }

        # 1. Early & Late Probability
        early_prob = float(np.mean(probs[:min(4, T)]))
        late_prob = float(np.mean(probs[max(0, T - 4):]))
        delta_p = late_prob - early_prob

        # 2. Dynamic Velocities (Slopes) across time
        velocities = []
        for t in range(T):
            t_start = max(0, t - self.k_lookback)
            dt = (t - t_start) * self.dt_frame
            if dt > 0:
                s = (probs[t] - probs[t_start]) / dt
            else:
                s = 0.0
            velocities.append(float(s))

        current_velocity = velocities[-1]

        # 3. Dynamic Acceleration (Curvature)
        if len(velocities) >= 2:
            current_acceleration = float((velocities[-1] - velocities[-2]) / self.dt_frame)
        else:
            current_acceleration = 0.0

        # 4. Predictive Lead-Time Metric (Delta t_lead in seconds)
        current_prob = float(probs[-1])
        if future_prob is None:
            future_prob = current_prob

        lead_time_seconds: Optional[float] = None
        if current_prob >= self.tau_crit:
            lead_time_seconds = 0.0  # Violence already at critical threshold
        elif current_velocity > 0.02:
            # Time required to cross critical threshold at current rate of escalation
            raw_lead_time = (self.tau_crit - current_prob) / current_velocity
            lead_time_seconds = float(np.clip(raw_lead_time, 0.2, 10.0))
        else:
            # Stable or negative velocity
            lead_time_seconds = None

        # 5. Composite Multi-Modal Hazard Index H(t)
        # H(t) = 0.45*P_future + 0.25*ReLU(Velocity) + 0.20*Weapon + 0.10*Audio
        pos_velocity = float(np.clip(current_velocity, 0.0, 1.0))
        hazard_index = float(
            0.45 * future_prob +
            0.25 * pos_velocity +
            0.20 * weapon_threat +
            0.10 * audio_anomaly
        )
        hazard_index = float(np.clip(hazard_index, 0.0, 1.0))

        # 6. Crossing Step Alert: First timestep where hazard >= tau_alert
        crossing_step: Optional[int] = None
        hazard_trajectory = []
        for t in range(T):
            step_v = max(0.0, velocities[t])
            step_h = float(0.50 * probs[t] + 0.25 * step_v + 0.15 * weapon_threat + 0.10 * audio_anomaly)
            hazard_trajectory.append(round(step_h, 3))
            if crossing_step is None and step_h >= self.tau_alert:
                crossing_step = t + 1  # 1-indexed frame step

        # 7. Escalation Behavioral Pattern Classification
        is_strictly_increasing = bool(np.all(np.diff(probs) >= -0.05) and delta_p > 0.15)

        if early_prob > 0.60:
            pattern = "EARLY_CRITICAL"
        elif is_strictly_increasing and current_prob >= self.tau_alert:
            pattern = "STRICT_ESCALATION"
        elif current_velocity > 0.10 and current_prob >= self.tau_alert:
            pattern = "SUDDEN_SPIKE"
        elif delta_p > 0.20:
            pattern = "GRADUAL_BUILDUP"
        elif current_prob >= self.tau_alert:
            pattern = "TRANSIENT_DISTURBANCE"
        else:
            pattern = "STABLE_NONVIOLENT"

        early_warning_alert = bool(
            hazard_index >= self.tau_alert or
            (lead_time_seconds is not None and lead_time_seconds <= 4.0) or
            weapon_threat > 0.50
        )

        return {
            "trajectory": [round(float(p), 4) for p in probs],
            "hazard_trajectory": hazard_trajectory,
            "current_probability": round(current_prob, 4),
            "future_violence_prob": round(float(future_prob), 4),
            "early_prob": round(early_prob, 4),
            "late_prob": round(late_prob, 4),
            "escalation_slope": round(current_velocity, 4),
            "escalation_acceleration": round(current_acceleration, 4),
            "lead_time_seconds": round(lead_time_seconds, 2) if lead_time_seconds is not None else None,
            "hazard_index": round(hazard_index, 4),
            "crossing_step": crossing_step,
            "pattern": pattern,
            "is_strictly_increasing": is_strictly_increasing,
            "early_warning_alert": early_warning_alert,
            "threat_level": (
                "CRITICAL" if hazard_index >= 0.75
                else "ELEVATED" if hazard_index >= 0.50
                else "ADVISORY" if hazard_index >= 0.30
                else "NORMAL"
            )
        }
