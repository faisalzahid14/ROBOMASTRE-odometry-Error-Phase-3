"""AprilTag approach with a camera-referenced distance estimate and tape error."""
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from robomaster import robot

from detect_tag import detect

START_M = 1.50
STOP_M = 0.10
STEP_M = 0.10
TOLERANCE_M = 0.025
OUTPUT = Path("phase3_10cm_measurements.csv")


def read_tag(camera, timeout):
    deadline = time.monotonic() + timeout
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
        x = np.asarray(corners[int(matches[0])]).reshape(4, 2)[:, 0]
        if np.ptp(x) >= 12:
            return float(x.mean() / frame.shape[1]), float(np.ptp(x))
    return None


def approach(ep):
    camera, chassis = ep.camera, ep.chassis
    positions = deque(maxlen=1)

    def on_position(data):
        x, y = map(float, data[:2])
        if math.isfinite(x) and math.isfinite(y):
            positions.append((time.monotonic(), x, y))

    def read_position(after):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            sample = positions[0] if positions else None
            if sample is not None and sample[0] >= after:
                return sample[1:]
            time.sleep(0.02)
        raise RuntimeError("No fresh chassis odometry.")

    streaming = subscribed = False
    try:
        streaming = bool(camera.start_video_stream(display=False))
        if not streaming:
            raise RuntimeError("Camera stream did not start.")
        tag = read_tag(camera, timeout=5)
        if tag is None:
            raise RuntimeError("AprilTag 25h9 ID 3 was not detected.")
        centre, start_width = tag
        print(f"AprilTag detected at {centre:.1%} of image width.", flush=True)
        if abs(centre - 0.5) > 0.10:
            raise RuntimeError("Centre the tag in the camera image and retry.")

        subscribed = bool(chassis.sub_position(cs=0, freq=20, callback=on_position))
        if not subscribed:
            raise RuntimeError("Chassis odometry subscription failed.")
        start_x, start_y = read_position(time.monotonic())

        estimate = odometry = START_M
        variance = 0.0
        q, r = 0.02 ** 2, 0.08 ** 2
        forward = sideways = 0.0
        updates = 0
        stop_reason = "step_limit"

        for step in range(1, 31):
            travel_left = min(estimate - STOP_M, START_M - STOP_M - forward)
            travel = min(0.05 if travel_left < 0.25 else STEP_M, max(0.05, travel_left))
            action = chassis.move(x=travel, y=0, z=0, xy_speed=0.5)
            completed = action.wait_for_completed(timeout=8)
            if not completed or getattr(action, "has_succeeded", True) is False:
                raise RuntimeError(f"Move {step} was not confirmed.")

            x, y = read_position(time.monotonic() + 0.25)
            forward, sideways = x - start_x, y - start_y
            new_odometry = math.hypot(START_M - forward, sideways)

            # Predict using the change in odometry range since the last step.
            estimate += new_odometry - odometry
            odometry = new_odometry
            variance += q

            tag = read_tag(camera, timeout=2)
            if tag is None:
                raise RuntimeError("Tag lost during approach.")
            camera_range = None
            if tag is not None:
                camera_range = START_M * start_width / tag[1]
                # Reject large jumps in apparent width before correcting the estimate.
                if abs(camera_range - estimate) <= 0.30:
                    gain = variance / (variance + r)
                    estimate += gain * (camera_range - estimate)
                    variance *= 1 - gain
                    updates += 1

            camera_text = "--" if camera_range is None else f"{camera_range:.3f} m"
            print(f"Step {step}: estimate {estimate:.3f} m | camera {camera_text} | "
                  f"sideways {sideways:+.3f} m", flush=True)

            if estimate <= STOP_M + TOLERANCE_M:
                stop_reason = "estimated_range"
                break
            if forward >= START_M - STOP_M - TOLERANCE_M:
                stop_reason = "forward_odometry"
                break

        return estimate, odometry, forward, sideways, updates, stop_reason
    finally:
        for active, close in (
            (subscribed, chassis.unsub_position),
            (streaming, camera.stop_video_stream),
        ):
            if active:
                try:
                    close()
                except Exception as error:
                    print(f"SDK cleanup: {error}")


def main():
    ep = robot.Robot()
    ep.initialize(conn_type="sta", proto_type="tcp")
    try:
        result = approach(ep)
    except RuntimeError as error:
        print(f"Trial incomplete: {error}")
        return
    finally:
        try:
            ep.close()
        except Exception as error:
            print(f"SDK close: {error}")

    estimate, odometry, forward, sideways, updates, reason = result
    if reason == "step_limit" or updates == 0:
        print(f"Trial incomplete: stop={reason}, camera corrections={updates}.")
        return

    print(f"Stopped. Estimated camera-to-tag distance: {estimate:.3f} m.")
    while True:
        text = input("Tape distance from camera lens to tag centre (m): ").strip()
        try:
            measured = float(text)
            if math.isfinite(measured) and measured > 0:
                break
        except ValueError:
            pass
        print("Enter a positive distance in metres, for example 0.76.")

    error = estimate - measured
    with OUTPUT.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow([
                "time", "start_m", "target_stop_m", "estimated_m", "measured_m",
                "signed_error_m", "absolute_error_m", "odometry_only_m",
                "forward_m", "sideways_m", "camera_corrections", "stop_reason",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), START_M, STOP_M,
            f"{estimate:.4f}", measured, f"{error:.4f}", f"{abs(error):.4f}",
            f"{odometry:.4f}", f"{forward:.4f}", f"{sideways:.4f}", updates, reason,
        ])
    print(f"Error: {error:+.3f} m | absolute error: {abs(error):.3f} m | saved to {OUTPUT}")
    with OUTPUT.open(newline="") as stream:
        errors = [float(row["signed_error_m"]) for row in csv.DictReader(stream)]
    mae = sum(map(abs, errors)) / len(errors)
    print(f"Trials: {len(errors)} | MAE: {mae:.3f} m | "
          f"signed error range: {min(errors):+.3f} to {max(errors):+.3f} m")


if __name__ == "__main__":
    main()
