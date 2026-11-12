import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np

from approach_apriltag import find_tag
from capture_phase3 import WheelRecorder, measure_distance, save_wheel_data
from phase3 import Filter, RPM_PROJECTION, connect, read_config

START_M = 1.50
STOP_M = 0.60
STEP_M = 0.10
STOP_TAG_WIDTH = 0.07
MAX_STEPS = 12


def estimate_now(recorder, started, config):
    with recorder.lock:
        samples = list(recorder.active)
        now = time.monotonic()

    times = [row[0] - started for row in samples]
    if len(times) < 5 or max(np.diff([0.0, *times, now - started])) > 0.5:
        raise ValueError("Missing wheel telemetry")

    scale = config["wheel_radius_m"] * 2 * math.pi / 60
    scales = scale * np.array([
        config["forward_sign"],
        config["yaw_sign"] / config["half_length_plus_width_m"],
    ])
    ekf = Filter(config)
    for ts, (_, *rpm) in zip(times, samples):
        reading = (RPM_PROJECTION @ np.asarray(rpm)) * scales
        ekf.observe(ts, "wheel", reading)

    return ekf.distance(), float(ekf.wheel_pose[0])


def approach(ep, recorder, started, config):
    previous_x = 0.0

    for step in range(MAX_STEPS + 1):
        tag = find_tag(ep.camera)
        if tag is None:
            return "tag_lost"

        centre, width = tag
        if abs(centre - 0.5) > 0.10:
            return "tag_off_centre"

        remaining, wheel_x = estimate_now(recorder, started, config)
        print(f"EKF range {remaining:.3f} m | tag width {width:.1%}", flush=True)

        if step and wheel_x - previous_x < 0.025:
            return "insufficient_forward_motion"
        if remaining <= STOP_M + STEP_M / 2:
            return "distance_threshold_reached"
        if width >= STOP_TAG_WIDTH:
            return "tag_size_threshold_reached"
        if step == MAX_STEPS:
            return "travel_limit_reached"

        previous_x = wheel_x
        print(f"Move {step + 1}: forward {STEP_M:.2f} m", flush=True)
        action = ep.chassis.move(x=STEP_M, y=0, z=0, xy_speed=0.5)
        if not action.wait_for_completed(timeout=8):
            return "move_unconfirmed"
        if getattr(action, "has_succeeded", True) is False:
            return "move_failed"
        time.sleep(0.4)


def main():
    config = read_config(Path("config.json"))
    config["start_m"] = START_M
    root = Path("phase3_trials")
    folder = root / datetime.now().strftime("trial_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    summary = {"trial": folder.name, "start_m": START_M, "target_stop_m": STOP_M}

    recorder = WheelRecorder()
    ep = connect("sta")
    streaming = subscribed = False
    started = None

    try:
        streaming = bool(ep.camera.start_video_stream(display=False))
        if not streaming:
            raise RuntimeError("Camera stream did not start")

        subscribed = bool(ep.chassis.sub_esc(freq=20, callback=recorder.on_esc))
        if not subscribed:
            raise RuntimeError("Wheel telemetry did not start")

        time.sleep(0.5)
        started = recorder.start()
        time.sleep(0.4)
        summary["status"] = approach(ep, recorder, started, config)

    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as error:
        summary.update(status="capture_error", detail=str(error))
    finally:
        try:
            if started is not None:
                ended, samples = recorder.stop()
                try:
                    estimate, wheel = save_wheel_data(
                        folder, samples, started, ended, config
                    )
                    summary.update(estimate_m=estimate, wheel_only_m=wheel)
                except ValueError as error:
                    summary["telemetry_error"] = str(error)
        finally:
            (folder / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            try:
                if subscribed:
                    ep.chassis.unsub_esc()
            finally:
                try:
                    if streaming:
                        ep.camera.stop_video_stream()
                finally:
                    ep.close()

    print(f"Run ended: {summary['status']}")
    if "estimate_m" not in summary:
        print(f"No usable estimate. See {folder / 'summary.json'}")
        return

    print(f"EKF final distance: {summary['estimate_m']:.3f} m")
    print("Once the robot is fully stopped, measure from the chassis centre "
          "to the tag centre along the floor.")
    measured = measure_distance()
    if measured is None:
        print(f"Wheel data and estimate saved in {folder}")
        return

    summary["measured_m"] = measured
    summary["error_m"] = summary["estimate_m"] - measured
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = ["trial", "status", "start_m", "target_stop_m", "estimate_m",
              "wheel_only_m", "measured_m", "error_m"]
    table = root / "measurements.csv"
    with table.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Saved to {table} | signed error {summary['error_m']:+.3f} m")


if __name__ == "__main__":
    main()
