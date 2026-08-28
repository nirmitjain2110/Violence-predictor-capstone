"""
End-to-End Video Analysis Test Script
======================================
Creates a sample synthetic video stream (with simulated escalating visual motion),
runs it through the full MultiModalViolencePipeline, and displays the complete
predictive metrics: P(V_{t+Delta t}), escalation slope S(t), lead-time Delta t,
crossing step, and annotated video generation.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.pipeline import MultiModalViolencePipeline


def create_synthetic_test_video(output_path: Path, num_frames: int = 30, fps: int = 10):
    """Creates a sample test video with gradual motion buildup to test escalation trajectory."""
    h, w = 360, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for t in range(num_frames):
        # Create background
        frame = np.ones((h, w, 3), dtype=np.uint8) * 30

        # Simulate escalating movement: circle grows and moves faster over time
        intensity = int((t / num_frames) * 200)
        radius = int(20 + (t / num_frames) * 60)
        cx = int(w/2 + np.sin(t * 0.5) * (t * 5))
        cy = int(h/2 + np.cos(t * 0.5) * (t * 4))

        cv2.circle(frame, (cx, cy), radius, (50, intensity, 255 - intensity), -1)
        cv2.putText(
            frame, f"TEST FEED - FRAME {t+1}/{num_frames}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )
        writer.write(frame)

    writer.release()
    print(f"[Synthetic Video] Generated test video: {output_path} ({num_frames} frames)")


def run_test():
    print("================================================================")
    print("  RUNNING MULTI-MODAL END-TO-END PIPELINE TEST")
    print("================================================================")

    test_video_path = APP_DIR / "sample_test_feed.mp4"
    create_synthetic_test_video(test_video_path, num_frames=32, fps=10)

    print("\n[1] Initializing MultiModalViolencePipeline...")
    pipeline = MultiModalViolencePipeline()

    print("\n[2] Executing Full Multi-Modal Analysis...")
    result = pipeline.analyze_video(test_video_path, weapon_sample_every=4)

    if "error" in result:
        print("ERROR:", result["error"])
        return

    print("\n--- Pipeline Analysis Results ---")
    print(f"Classification Label        : {result['label']}")
    print(f"Current Violence Probability: {result['violence_probability']:.2%}")
    print(f"Future Violence Probability : {result['future_violence_probability']:.2%}")
    print(f"Audio Stream Used           : {result['audio_used']} (Energy: {result['audio_energy_score']})")
    print(f"Weapons Detected            : {result['weapons_detected']}")
    print(f"6D Threat Vector            : {result['weapon_threat_vector']}")

    m = result.get("metrics", {})
    print("\n--- Dynamic Early Warning & Lead-Time Metrics ---")
    print(f"Escalation Pattern          : {m.get('pattern')}")
    print(f"Escalation Velocity S(t)    : {m.get('escalation_slope')} /s")
    print(f"Escalation Acceleration A(t): {m.get('escalation_acceleration')}")
    print(f"Predictive Lead-Time (Δt)   : {m.get('lead_time_seconds')} seconds")
    print(f"Composite Hazard Index H(t) : {m.get('hazard_index')}")
    print(f"Crossing Step (t*)          : Frame {m.get('crossing_step')}")
    print(f"Early Warning Triggered     : {m.get('early_warning_alert')}")
    print(f"Threat Level                : {m.get('threat_level')}")
    print(f"Temporal Trajectory P(1:16) : {m.get('trajectory')}")

    # Generate Annotated Output Video
    annotated_output_path = APP_DIR / "sample_test_feed_annotated.mp4"
    print(f"\n[3] Rendering Annotated Video with Telemetry HUD to: {annotated_output_path}...")
    pipeline.annotate_video_frames(
        all_frames=result["all_frames"],
        fps=result["fps"],
        analysis_result=result,
        output_path=annotated_output_path
    )
    print("Annotated video saved successfully!")

    print("\n================================================================")
    print("  ALL END-TO-END TESTS COMPLETED SUCCESSFULLY!")
    print("================================================================")


if __name__ == "__main__":
    run_test()
