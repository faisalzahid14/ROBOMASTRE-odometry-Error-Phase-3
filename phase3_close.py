"""Collect Phase 3 trials from 1.50 m to approximately 0.10 m."""
import argparse
import csv
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty
from statistics import median

import cv2
import numpy as np
from robomaster import robot

from detect_tag import detect

START_M = 1.50
TARGET_M = 0.10
STOP_THRESHOLD_M = 0.115
REFERENCE = Path("phase3_close_reference.json")
RESULTS = Path("phase3_close_measurements.csv")


def read_tag(camera, timeout=3):
    deadline = time.monotonic() + timeout
    centres, widths = [], []
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
        matches = np.flatnonzero(np.asarray(ids).ravel() == 3)
        if len(matches) != 1:
            continue

        points = np.asarray(corners[int(matches[0])]).reshape(4, 2)
        height, width = frame.shape[:2]
        if (np.any(points[:, 0] < 3) or np.any(points[:, 0] > width - 4)
                or np.any(points[:, 1] < 3) or np.any(points[:, 1] > height - 4)):
            continue
        sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        tag_width = (sides[0] + sides[2]) / 2
        if tag_width < 12:
            continue
        centres.append(float(points[:, 0].mean() / width))
        widths.append(float(tag_width / width))
        if len(widths) == 3:
            return median(centres), median(widths)
    return None


def set_reference(camera):
    tag = read_tag(camera, timeout=6)
    if tag is None:
        raise RuntimeError("The camera cannot see the full tag at 10 cm. "
                           "Align the tag with the lens and retry.")
    centre, width = tag
    if abs(centre - 0.5) > 0.15:
        raise RuntimeError("Centre the tag in the image before saving the 10 cm reference.")
    REFERENCE.write_text(json.dumps({"width_fraction_at_10cm": width}, indent=2) + "\n")
    print(f"Saved the 10 cm reference ({width:.1%} of image width). No movement commanded.")


def read_position(positions, after):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        sample = positions[0] if positions else None
        if sample is not None and sample[0] >= after:
            return sample[1:]
        time.sleep(0.02)
    raise RuntimeError("No fresh chassis position reading.")


def approach(ep, close_width):
    camera, chassis = ep.camera, ep.chassis
    tag = read_tag(camera, timeout=6)
    if tag is None:
        raise RuntimeError("AprilTag 25h9 ID 3 was not detected at the 1.50 m mark.")
    centre, start_width = tag
    if abs(centre - 0.5) > 0.10:
        raise RuntimeError(f"Tag centre is at {centre:.1%}. Aim the chassis and camera at it.")
    if close_width <= start_width * 1.5:
        raise RuntimeError("The 10 cm reference is inconsistent with the starting image.")

    inverse_width_span = 1 / start_width - 1 / close_width
    positions = deque(maxlen=1)

    def on_position(data):
        x, y = map(float, data[:2])
        if math.isfinite(x) and math.isfinite(y):
            positions.append((time.monotonic(), x, y))

    subscribed = chassis.sub_position(cs=0, freq=20, callback=on_position)
    if not subscribed:
        raise RuntimeError("Chassis position subscription failed.")

    try:
        x0, y0 = read_position(positions, time.monotonic() + 0.6)
        estimate = camera_range = odometry_range = START_M
        variance = 0.0
        forward = 0.0
        print("AprilTag detected. Starting the 1.50 m to 0.10 m trial.", flush=True)

        for step in range(1, 41):
            if camera_range > 0.35:
                travel = 0.10
            elif camera_range > 0.20:
                travel = 0.05
            else:
                travel = min(0.03, max(0.02, camera_range - TARGET_M))

            action = chassis.move(x=travel, y=0, z=0, xy_speed=0.5)
            completed = action.wait_for_completed(timeout=8)
            if not completed or getattr(action, "has_succeeded", True) is False:
                raise RuntimeError(f"Move {step} was not confirmed.")

            x, y = read_position(positions, time.monotonic() + 0.25)
            next_forward, sideways = x - x0, y - y0
            if next_forward - forward < 0.005:
                raise RuntimeError(f"Move {step} produced less than 5 mm of forward odometry.")
            forward = next_forward
            new_odometry = math.hypot(START_M - forward, sideways)
            estimate += new_odometry - odometry_range
            odometry_range = new_odometry

            tag = read_tag(camera)
            if tag is None:
                raise RuntimeError(f"Tag lost after move {step}. Last camera range was "
                                   f"{camera_range:.3f} m. Check the tag height and alignment.")
            _, width = tag
            camera_range = TARGET_M + (START_M - TARGET_M) * (
                (1 / width - 1 / close_width) / inverse_width_span
            )
            variance += (0.015 + 0.1 * travel) ** 2
            if abs(camera_range - estimate) <= 0.30:
                camera_sd = 0.025 if camera_range < 0.35 else 0.08
                gain = variance / (variance + camera_sd ** 2)
                estimate += gain * (camera_range - estimate)
                variance *= 1 - gain
            else:
                print(f"Step {step}: skipped camera correction "
                      f"(camera {camera_range:.3f} m, predicted {estimate:.3f} m)",
                      flush=True)

            print(f"Step {step}: camera {camera_range:.3f} m | "
                  f"estimate {estimate:.3f} m | sideways {sideways:+.3f} m", flush=True)
            if camera_range <= STOP_THRESHOLD_M:
                return estimate, camera_range, odometry_range, forward, sideways
            if forward >= START_M - 0.04:
                raise RuntimeError("Forward travel limit reached before the camera stop point.")

        raise RuntimeError("Step limit reached before the camera stop point.")
    finally:
        try:
            chassis.unsub_position()
        except Exception as error:
            print(f"SDK cleanup: {error}")


def tape_distance():
    while True:
        entry = input("Tape distance from lens to tag centre in metres (Enter to skip): ").strip()
        if not entry:
            return None
        try:
            measured = float(entry)
        except ValueError:
            measured = 0
        if math.isfinite(measured) and measured > 0:
            return measured
        print("Enter a positive distance, for example 0.11.")


def save_result(result, measured):
    estimate, camera_range, odometry, forward, sideways = result
    error = estimate - measured
    with RESULTS.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(["time", "start_m", "target_m", "estimated_m", "measured_m",
                             "error_m", "absolute_error_m", "camera_m", "odometry_m",
                             "forward_m", "sideways_m"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), START_M, TARGET_M,
                         f"{estimate:.4f}", measured, f"{error:.4f}", f"{abs(error):.4f}",
                         f"{camera_range:.4f}", f"{odometry:.4f}", f"{forward:.4f}",
                         f"{sideways:.4f}"])
    with RESULTS.open(newline="") as stream:
        errors = [float(row["error_m"]) for row in csv.DictReader(stream)]
    mae = sum(map(abs, errors)) / len(errors)
    print(f"Trial {len(errors)} saved to {RESULTS}. Error {error:+.3f} m | MAE {mae:.3f} m | "
          f"signed error range {min(errors):+.3f} to {max(errors):+.3f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-10cm", action="store_true",
                        help="Record a stationary camera image at a tape-measured 0.10 m.")
    args = parser.parse_args()
    close_width = None
    if not args.set_10cm:
        if not REFERENCE.exists():
            parser.error("First run python phase3_close.py --set-10cm at a measured 10 cm.")
        close_width = float(json.loads(REFERENCE.read_text())["width_fraction_at_10cm"])

    ep = robot.Robot()
    streaming = False
    result = None
    try:
        ep.initialize(conn_type="sta", proto_type="tcp")
        streaming = bool(ep.camera.start_video_stream(display=False))
        if not streaming:
            raise RuntimeError("Camera stream did not start.")
        if args.set_10cm:
            set_reference(ep.camera)
        else:
            result = approach(ep, close_width)
    except (RuntimeError, TimeoutError, TypeError, AttributeError, OSError) as error:
        print(f"Run incomplete: {error}")
    finally:
        if streaming:
            try:
                ep.camera.stop_video_stream()
            except Exception as error:
                print(f"SDK cleanup: {error}")
        try:
            ep.close()
        except Exception as error:
            print(f"SDK close: {error}")

    if result is not None:
        print(f"Stopped. Estimated lens-to-tag distance: {result[0]:.3f} m.")
        measured = tape_distance()
        if measured is not None:
            save_result(result, measured)


if __name__ == "__main__":
    main()
