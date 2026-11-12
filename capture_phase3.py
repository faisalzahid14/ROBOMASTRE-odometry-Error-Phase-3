"""Record up to ten Phase 3 manual runs using the existing EKF and AprilTag detector."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from queue import Empty
from threading import Lock
import time

import numpy as np

from phase3 import AprilTagLock, FIELDS, Filter, RPM_PROJECTION, connect, read_config


class WheelRecorder:
    def __init__(self):
        self.lock = Lock()
        self.last_packet = None
        self.active = None

    def on_esc(self, data):
        rpm = np.asarray(data[0], dtype=float)
        if rpm.shape != (4,) or not np.isfinite(rpm).all():
            return
        with self.lock:
            now = time.monotonic()
            self.last_packet = now
            if self.active is not None:
                self.active.append((now, *rpm))

    def start(self):
        with self.lock:
            now = time.monotonic()
            if self.last_packet is None or now - self.last_packet > 0.5:
                raise RuntimeError("No recent wheel telemetry. Check the robot connection.")
            self.active = []
            return now

    def stop(self):
        with self.lock:
            samples, self.active = self.active, None
            return time.monotonic(), samples


def find_tag(camera, detector):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            frame = camera.read_cv2_image(timeout=0.5, strategy="newest")
        except (Empty, TimeoutError):
            continue
        if frame is not None:
            offset = detector.observe(frame)
            if offset is not None and abs(offset) <= 0.12:
                return frame, offset
    return None, None


def save_wheel_data(folder, samples, started, ended, config):
    times = [stamp - started for stamp, *_ in samples]
    with (folder / "wheel_rpm.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "rpm_1", "rpm_2", "rpm_3", "rpm_4"])
        writer.writerows((ts, *row[1:]) for ts, row in zip(times, samples))

    gaps = np.diff([0.0, *times, ended - started])
    if len(samples) < 10 or max(gaps) > 0.5:
        raise ValueError("Wheel telemetry was missing or interrupted.")

    scales = config["wheel_radius_m"] * 2 * math.pi / 60 * np.array([
        config["forward_sign"], config["yaw_sign"] / config["half_length_plus_width_m"]
    ])
    estimate = Filter(config)
    with (folder / "events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for ts, (_, *rpm) in zip(times, samples):
            reading = (RPM_PROJECTION @ np.asarray(rpm)) * scales
            innovation, accepted = estimate.observe(ts, "wheel", reading)
            writer.writerow(estimate.row(ts, "wheel", reading, innovation, accepted))

    forward_m = float(estimate.wheel_pose[0])
    if forward_m < 0.02:
        raise ValueError(f"Wheel odometry recorded only {forward_m:+.3f} m forward. "
                         "Check the raw RPMs and the configured forward sign.")
    if forward_m >= config["start_m"]:
        raise ValueError("Wheel odometry places the robot at or beyond the target.")

    wheel_range = float(np.linalg.norm(estimate.marker - estimate.wheel_pose[:2]))
    return estimate.distance(), wheel_range


def measure_distance():
    while True:
        entry = input("Measured final distance in metres (e.g. 0.52), or q to skip: ").strip()
        if entry.lower() == "q":
            return None
        try:
            distance = float(entry)
        except ValueError:
            distance = float("nan")
        if math.isfinite(distance) and distance > 0:
            return distance
        print("Enter a positive distance in metres.")


def collect(args, ep, config, output, completed, table):
    import cv2

    recorder = WheelRecorder()
    detector = AprilTagLock(config)
    camera = ep.camera
    subscribed = streaming = False
    try:
        subscribed = bool(ep.chassis.sub_esc(freq=20, callback=recorder.on_esc))
        if not subscribed:
            raise RuntimeError("The robot did not accept the wheel telemetry subscription.")
        streaming = bool(camera.start_video_stream(display=False))
        if not streaming:
            raise RuntimeError("The robot did not start the camera stream.")

        with table.open("a", newline="") as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(["trial", "attempt", "control", "start_m", "estimate_m",
                                 "wheel_only_m", "measured_m", "samples"])
                stream.flush()

            while completed < args.trials:
                print(f"\nTrial {completed + 1}/{args.trials}. "
                      f"Reset to {config['start_m']:.2f} m and face the tag.")
                if input("Enter to check the tag, or q to finish: ").strip().lower() == "q":
                    break
                frame, offset = find_tag(camera, detector)
                if frame is None:
                    print("Configured AprilTag was not centred. Align the robot and try again.")
                    continue

                attempt = datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
                folder = output / attempt
                folder.mkdir()
                if not cv2.imwrite(str(folder / "start_tag.jpg"), frame):
                    raise OSError("Could not save the AprilTag image.")
                summary = {"status": "capture_started", "control": "manual",
                           "camera_mode": "start_tag_lock_only", "tag_offset": offset,
                           "start_m": config["start_m"]}
                summary_path = folder / "summary.json"
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")

                started = recorder.start()
                input(f"RECORDING. Move straight toward the tag manually. "
                      f"Stop near {config['stop_m']:.2f} m, then press Enter: ")
                ended, samples = recorder.stop()
                summary["samples"] = len(samples)
                try:
                    ekf_range, wheel_range = save_wheel_data(folder, samples, started, ended, config)
                except ValueError as error:
                    summary.update(status="invalid_capture", reason=str(error))
                    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                    print(f"{error} Raw data saved in {folder}. Retry this trial.")
                    continue

                summary.update(status="awaiting_measurement", estimate_m=ekf_range,
                               wheel_only_m=wheel_range)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                print("Measure from the floor projection of the chassis centre "
                      "to the floor projection of the tag centre.")
                distance = measure_distance()
                if distance is None:
                    break
                summary.update(status="manual_trial_recorded", measured_m=distance)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                completed += 1
                writer.writerow([completed, attempt, "manual", config["start_m"],
                                 f"{ekf_range:.6f}", f"{wheel_range:.6f}",
                                 distance, len(samples)])
                stream.flush()
                print(f"Saved {completed}/{args.trials} to {table}.")
    finally:
        try:
            if streaming:
                camera.stop_video_stream()
        finally:
            if subscribed:
                ep.chassis.unsub_esc()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("manual_capture"))
    parser.add_argument("--connection", choices=["ap", "sta", "rndis"], default="sta")
    parser.add_argument("--trials", type=int, choices=range(1, 11), default=10)
    args = parser.parse_args()

    config = read_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    saved_config = args.output / "config.json"
    if saved_config.exists():
        if json.loads(saved_config.read_text()) != config:
            parser.error("Settings changed. Use a new --output folder.")
    else:
        saved_config.write_text(json.dumps(config, indent=2) + "\n")
    table = args.output / "measurements.csv"
    completed = 0
    if table.exists():
        with table.open(newline="") as stream:
            completed = sum(1 for _ in csv.DictReader(stream))
    if completed >= args.trials:
        print(f"{completed} trials already recorded in {table}.")
        return

    print("Manual data capture. This script sends no drive commands.")
    ep = connect(args.connection)
    try:
        collect(args, ep, config, args.output, completed, table)
    except (KeyboardInterrupt, EOFError):
        print(f"\nStopped. Previously recorded trials are in {table}.")
    finally:
        ep.close()


if __name__ == "__main__":
    main()
