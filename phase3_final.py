"""Run one RoboMaster Phase 3 trial and save its sensor data and measured error."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from queue import Empty
from threading import Lock
import time

import cv2
import numpy as np

from detect_tag import detect
from phase3 import FIELDS, Filter, RPM_PROJECTION, connect, read_config

STEP_M = 0.10
MAX_STEPS = 18
MAX_LATERAL_M = 0.08
CENTER_TOLERANCE = 0.15
VALID_STOPS = {"stop_distance_reached", "camera_clearance_reached"}


class Telemetry:
    def __init__(self):
        self.lock = Lock()
        self.position = None
        self.wheels = []

    def on_position(self, data):
        x, y = map(float, data[:2])
        if math.isfinite(x) and math.isfinite(y):
            with self.lock:
                self.position = time.monotonic(), x, y

    def on_wheels(self, data):
        rpm = np.asarray(data[0], dtype=float)
        if rpm.shape == (4,) and np.isfinite(rpm).all():
            with self.lock:
                self.wheels.append((time.monotonic(), rpm))

    def reset(self):
        with self.lock:
            self.wheels.clear()
            return time.monotonic()

    def snapshot(self):
        with self.lock:
            wheels, self.wheels = self.wheels, []
            return self.position, wheels


def find_tag(camera):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            frame = camera.read_cv2_image(timeout=0.25, strategy="newest")
        except (Empty, TimeoutError):
            continue
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(gray, "25h9")
        if ids is None:
            continue
        match = np.flatnonzero(np.asarray(ids).ravel() == 3)
        if len(match) != 1:
            continue
        points = np.asarray(corners[int(match[0])]).reshape(4, 2)
        width = np.ptp(points[:, 0]) / frame.shape[1]
        if width >= 0.01:
            return float(points[:, 0].mean() / frame.shape[1]), float(width), frame
    return None


def camera_measurement(centre, width, focal, cfg):
    # Pinhole approximation. The measured starting distance supplies image scale.
    depth = focal * cfg["tag_side_m"] / width
    forward = depth + cfg["camera_forward_m"]
    left = cfg["camera_left_m"] - depth * (centre - 0.5) / focal
    return np.array([math.hypot(forward, left), math.atan2(left, forward)])


def run_trial(ep, cfg, folder, summary):
    telemetry = Telemetry()
    camera = ep.camera
    chassis = ep.chassis
    position_on = wheels_on = camera_on = False
    ekf = Filter(cfg)
    scales = cfg["wheel_radius_m"] * 2 * math.pi / 60 * np.array([
        cfg["forward_sign"], cfg["yaw_sign"] / cfg["half_length_plus_width_m"]
    ])
    last_wheel = None
    stale_wheel = False
    accepted_updates = 0
    started = None

    def ingest(wheels, events, raw):
        nonlocal last_wheel, stale_wheel
        for stamp, rpm in wheels:
            if stamp < started:
                continue
            if stamp - (started if last_wheel is None else last_wheel) > 0.5:
                stale_wheel = True
            last_wheel = stamp
            ts = max(stamp - started, ekf.t)
            z = (RPM_PROJECTION @ rpm) * scales
            innovation, accepted = ekf.observe(ts, "wheel", z)
            raw.writerow([stamp - started, *rpm])
            events.writerow(ekf.row(ts, "wheel", z, innovation, accepted))

    with (folder / "events.csv").open("w", newline="") as event_file, (
        folder / "wheel_rpm.csv"
    ).open("w", newline="") as raw_file:
        events, raw = csv.writer(event_file), csv.writer(raw_file)
        events.writerow(FIELDS)
        raw.writerow(["time_s", "rpm_1", "rpm_2", "rpm_3", "rpm_4"])
        try:
            position_on = bool(chassis.sub_position(
                cs=0, freq=10, callback=telemetry.on_position
            ))
            wheels_on = bool(chassis.sub_esc(freq=20, callback=telemetry.on_wheels))
            camera_on = bool(camera.start_video_stream(display=False))
            if not (position_on and wheels_on and camera_on):
                raise RuntimeError("Could not start camera and chassis telemetry.")

            tag = find_tag(camera)
            print("Start tag: not detected" if tag is None else f"Start tag centre: {tag[0]:.1%}", flush=True)
            if tag is None or abs(tag[0] - 0.5) > 0.06:
                return "start_tag_missing_or_off_centre"
            centre, initial_width, frame = tag
            cv2.imwrite(str(folder / "start_tag.jpg"), frame)
            focal = initial_width * (cfg["start_m"] - cfg["camera_forward_m"]) / cfg["tag_side_m"]
            summary["initial_tag_width_fraction"] = initial_width
            summary["effective_focal_width_fraction"] = focal

            position, wheels = telemetry.snapshot()
            now = time.monotonic()
            if position is None or not wheels or now - min(position[0], wheels[-1][0]) > 0.5:
                return "no_initial_telemetry"

            _, start_x, start_y = position
            started = telemetry.reset()
            sideways = 0.0
            previous_forward = 0.0
            remaining = cfg["start_m"]

            for step in range(1, MAX_STEPS + 1):
                travel = min(STEP_M, max(0.05, remaining - cfg["stop_m"]))
                print(f"Move {step}: forward {travel:.2f} m", flush=True)
                action = chassis.move(x=travel, y=0, z=0, xy_speed=0.5)
                completed = action.wait_for_completed(timeout=8)
                if not completed or getattr(action, "has_succeeded", True) is False:
                    return "move_unconfirmed"

                time.sleep(0.4)
                tag = find_tag(camera)
                position, wheels = telemetry.snapshot()
                ingest(wheels, events, raw)
                now = time.monotonic()
                if (position is None or last_wheel is None or stale_wheel
                        or now - min(position[0], last_wheel) > 0.5):
                    return "telemetry_lost"

                _, x, y = position
                forward, sideways = x - start_x, y - start_y
                summary.update(completed_steps=step, chassis_forward_m=forward,
                               chassis_sideways_m=sideways)
                if abs(sideways) > MAX_LATERAL_M:
                    return "sideways_limit"
                if forward - previous_forward < 0.02:
                    return "insufficient_forward_motion"
                previous_forward = forward
                if tag is None:
                    return "tag_lost"
                centre, width, frame = tag
                print(f"Tag centre: {centre:.1%}", flush=True)
                if abs(centre - 0.5) > CENTER_TOLERANCE:
                    return "tag_off_centre"

                z = camera_measurement(centre, width, focal, cfg)
                ts = max(ekf.t, now - started)
                innovation, accepted = ekf.observe(ts, "tag", z)
                events.writerow(ekf.row(ts, "tag", z, innovation, accepted))
                accepted_updates += int(accepted)
                remaining = ekf.distance()
                summary["last_camera_range_m"] = float(z[0])
                print(f"EKF {remaining:.3f} m | camera approx. {z[0]:.3f} m | "
                      f"sideways {sideways:+.3f} m", flush=True)

                if remaining <= cfg["stop_m"] + 0.03 or z[0] <= cfg["stop_m"]:
                    cv2.imwrite(str(folder / "stop_tag.jpg"), frame)
                    if not accepted_updates:
                        return "no_camera_correction_accepted"
                    if remaining <= cfg["stop_m"] + 0.03:
                        return "stop_distance_reached"
                    return "camera_clearance_reached"
                if forward >= cfg["start_m"] - cfg["stop_m"] + 0.10:
                    return "odometry_travel_limit"
            return "step_limit"
        finally:
            try:
                if started is not None:
                    _, wheels = telemetry.snapshot()
                    ingest(wheels, events, raw)
                    summary.update(
                        estimate_m=ekf.distance(),
                        wheel_only_m=float(np.linalg.norm(ekf.marker - ekf.wheel_pose[:2])),
                        accepted_camera_updates=accepted_updates,
                    )
            finally:
                for active, stop in (
                    (position_on, chassis.unsub_position),
                    (wheels_on, chassis.unsub_esc),
                    (camera_on, camera.stop_video_stream),
                ):
                    if active:
                        try:
                            stop()
                        except Exception as error:
                            print(f"SDK shutdown: {error}", flush=True)


def read_tape():
    while True:
        entry = input("Tape distance in metres, or Enter to skip: ").strip()
        if not entry:
            return None
        try:
            value = float(entry)
        except ValueError:
            value = 0
        if math.isfinite(value) and value > 0:
            return value
        print("Enter a positive distance in metres.")


def save_summary(folder, summary):
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=1.50)
    parser.add_argument("--stop", type=float, default=0.60)
    parser.add_argument("--output", type=Path, default=Path("phase3_final_runs"))
    args = parser.parse_args()
    if not (1 <= args.start <= 2 and 0.50 <= args.stop <= args.start - 0.20):
        parser.error("Use a start of 1–2 m and a stop of at least 0.50 m, with 0.20 m travel.")

    cfg = read_config(Path("config.json"))
    cfg.update(start_m=args.start, stop_m=args.stop, tag_family="25h9", tag_id=3)
    # Apparent-width ranging needs more uncertainty than calibrated camera pose.
    cfg["tag_sd"] = [0.15, 0.10]

    args.output.mkdir(parents=True, exist_ok=True)
    settings = args.output / "config.json"
    if settings.exists() and json.loads(settings.read_text()) != cfg:
        parser.error("Start, stop or filter settings changed. Use a new --output directory.")
    settings.write_text(json.dumps(cfg, indent=2) + "\n")

    name = datetime.now().strftime("trial_%Y%m%d_%H%M%S_%f")
    folder = args.output / name
    folder.mkdir()
    summary = {"trial": name, "start_m": args.start, "target_stop_m": args.stop,
               "camera_mode": "apparent_width_with_measured_start_reference"}
    ep = None
    try:
        ep = connect("sta")
        summary["status"] = run_trial(ep, cfg, folder, summary)
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as error:
        summary.update(status="capture_error", detail=str(error))
    finally:
        save_summary(folder, summary)
        if ep is not None:
            try:
                ep.close()
            except Exception as error:
                print(f"SDK close: {error}", flush=True)

    print(f"Run ended: {summary['status']} | data: {folder}")
    if "detail" in summary:
        print(summary["detail"])
    if summary["status"] not in VALID_STOPS:
        return

    print(f"Final EKF range: {summary['estimate_m']:.3f} m")
    print("Measure along the floor from the chassis centre to the tag centre.")
    measured = read_tape()
    if measured is None:
        return
    summary.update(measured_m=measured, error_m=summary["estimate_m"] - measured)
    save_summary(folder, summary)
    fields = ["trial", "status", "start_m", "target_stop_m", "estimate_m",
              "wheel_only_m", "measured_m", "error_m", "accepted_camera_updates"]
    table = args.output / "measurements.csv"
    with table.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Saved {table} | signed error {summary['error_m']:+.3f} m")


if __name__ == "__main__":
    main()
