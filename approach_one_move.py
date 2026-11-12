import csv
from datetime import datetime
import json
from pathlib import Path
import time

from approach_apriltag import find_tag
from capture_phase3 import WheelRecorder, measure_distance, save_wheel_data
from phase3 import connect, read_config


def main():
    start_m = float(input("Measured starting distance in metres: "))
    stop_m = float(input("Desired stopping distance in metres: "))
    if not (1.0 <= start_m <= 2.0 and 0.35 <= stop_m < start_m):
        raise SystemExit("Use a start of 1–2 m and a stop of at least 0.35 m.")

    travel_m = round(start_m - stop_m, 3)
    if not 0.10 <= travel_m <= 1.50:
        raise SystemExit("The forward distance must be between 0.10 and 1.50 m.")

    config = read_config(Path("config.json"))
    config["start_m"] = start_m

    root = Path("one_move_runs")
    folder = root / datetime.now().strftime("trial_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    summary = {
        "status": "started",
        "start_m": start_m,
        "desired_stop_m": stop_m,
        "commanded_travel_m": travel_m,
    }
    recorder = WheelRecorder()
    ep = connect("sta")
    subscribed = False

    try:
        if not ep.camera.start_video_stream(display=False):
            raise RuntimeError("Camera stream did not start.")
        try:
            tag = find_tag(ep.camera)
        finally:
            ep.camera.stop_video_stream()

        if tag is None or abs(tag[0] - 0.5) > 0.10:
            summary["status"] = "tag_missing_or_off_centre"
            raise SystemExit("Aim the robot at the AprilTag and try again.")

        summary["tag_centre_fraction"], summary["tag_width_fraction"] = tag
        subscribed = bool(ep.chassis.sub_esc(freq=20, callback=recorder.on_esc))
        if not subscribed:
            raise RuntimeError("Wheel telemetry did not start.")
        time.sleep(0.5)

        print(f"AprilTag detected. Moving forward {travel_m:.3f} m once.", flush=True)
        started = recorder.start()
        summary["status"] = "move_interrupted"
        try:
            action = ep.chassis.move(x=travel_m, y=0, z=0, xy_speed=0.5)
            completed = action.wait_for_completed(timeout=15)
            succeeded = completed and getattr(action, "has_succeeded", True)
            summary["status"] = "move_completed" if succeeded else "move_unconfirmed"
        finally:
            time.sleep(0.3)
            ended, samples = recorder.stop()
            try:
                estimate, wheel = save_wheel_data(folder, samples, started, ended, config)
                summary.update(estimate_m=estimate, wheel_only_m=wheel)
            except ValueError as error:
                summary["telemetry_error"] = str(error)
    finally:
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        try:
            if subscribed:
                ep.chassis.unsub_esc()
        finally:
            ep.close()

    if summary["status"] != "move_completed" or "estimate_m" not in summary:
        print(f"Trial incomplete. Diagnostic data saved in {folder}.")
        return

    print(f"EKF final distance: {summary['estimate_m']:.3f} m")
    print("Measure from the chassis centre to the tag centre before moving either.")
    measured_m = measure_distance()
    if measured_m is None:
        print(f"Wheel data and estimate saved in {folder}.")
        return

    summary.update(status="recorded", measured_m=measured_m,
                   error_m=summary["estimate_m"] - measured_m)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = root / "measurements.csv"
    with table.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(["trial", "start_m", "desired_stop_m", "estimate_m",
                             "wheel_only_m", "measured_m", "error_m"])
        writer.writerow([folder.name, start_m, stop_m, summary["estimate_m"],
                         summary["wheel_only_m"], measured_m, summary["error_m"]])
    print(f"Measurement saved to {table}.")


if __name__ == "__main__":
    main()
