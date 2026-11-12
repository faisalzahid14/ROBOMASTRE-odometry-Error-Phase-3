"""Collect eight Phase 3 runs in one session."""
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
STEP_M = 0.10
STEPS = 14
TRIALS = 8
RESULTS = Path("phase3_eight_measurements.csv")
FIELDS = [
    "trial", "time", "start_m", "commanded_travel_m", "estimated_remaining_m",
    "measured_remaining_m", "signed_error_m", "absolute_error_m",
    "odometry_travel_m", "odometry_samples_used", "odometry_samples_skipped",
    "camera_corrections",
]


def tag_width(camera, timeout):
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
        points = np.asarray(corners[int(matches[0])]).reshape(4, 2)
        edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        width = float((edges[0] + edges[2]) / 2)
        if width >= 12:
            return width / frame.shape[1]
    return None


def fresh_position(positions, after):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        latest = positions[0] if positions else None
        if latest is not None and latest[0] >= after:
            return latest[1:]
        time.sleep(0.02)
    raise RuntimeError("No fresh chassis position reading.")


def run_trial(ep, positions):
    start_width = tag_width(ep.camera, timeout=5)
    if start_width is None:
        raise RuntimeError("AprilTag 25h9 ID 3 not detected at the starting mark.")
    print("AprilTag detected. Starting forward approach.", flush=True)

    # The position stream uses a fixed origin. Use differences between samples.
    previous = fresh_position(positions, time.monotonic() + 0.50)
    estimate = START_M
    variance = 0.0
    odometry_travel = 0.0
    odometry_used = odometry_skipped = corrections = 0

    for step in range(1, STEPS + 1):
        action = ep.chassis.move(x=STEP_M, y=0, z=0, xy_speed=0.5)
        completed = action.wait_for_completed(timeout=8)
        if not completed or getattr(action, "has_succeeded", True) is False:
            raise RuntimeError(f"Move {step} was not confirmed by the SDK.")

        current = fresh_position(positions, time.monotonic() + 0.30)
        delta = math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current

        if 0.005 <= delta <= STEP_M + 0.08:
            odometry_travel += delta
            estimate -= delta
            variance += 0.025 ** 2
            odometry_used += 1
        else:
            # Leave the predicted distance unchanged for an invalid packet.
            variance += 0.10 ** 2
            odometry_skipped += 1
            print(f"Step {step}: ignored odometry change of {delta:.3f} m.",
                  flush=True)

        width = tag_width(ep.camera, timeout=0.6)
        if width is not None:
            camera_range = START_M * start_width / width
            if abs(camera_range - estimate) <= 0.30:
                camera_variance = 0.10 ** 2
                gain = variance / (variance + camera_variance)
                estimate += gain * (camera_range - estimate)
                variance *= 1 - gain
                corrections += 1

        estimate = max(0.0, estimate)
        print(f"Move {step}/{STEPS} | estimate {estimate:.3f} m remaining", flush=True)

    if not odometry_used or not corrections:
        raise RuntimeError("The run needs usable odometry and at least one camera correction.")
    return {
        "estimated_remaining_m": estimate,
        "odometry_travel_m": odometry_travel,
        "odometry_samples_used": odometry_used,
        "odometry_samples_skipped": odometry_skipped,
        "camera_corrections": corrections,
    }


def tape_reading():
    while True:
        entry = input("Tape distance from camera lens to tag centre in metres (or q): ").strip()
        if entry.lower() == "q":
            return None
        try:
            reading = float(entry)
            if math.isfinite(reading) and reading > 0:
                return reading
        except ValueError:
            pass
        print("Enter metres, for example 0.18.")


def saved_errors():
    if not RESULTS.exists():
        return []
    with RESULTS.open(newline="") as stream:
        return [float(row["signed_error_m"]) for row in csv.DictReader(stream)]


def save_trial(number, result, measured):
    error = result["estimated_remaining_m"] - measured
    with RESULTS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({
            **result,
            "trial": number,
            "time": datetime.now().isoformat(timespec="seconds"),
            "start_m": START_M,
            "commanded_travel_m": round(STEPS * STEP_M, 2),
            "measured_remaining_m": measured,
            "signed_error_m": error,
            "absolute_error_m": abs(error),
        })
    print(f"Trial {number}/{TRIALS} saved | signed error {error:+.3f} m", flush=True)
    return error


def show_summary(errors):
    if errors:
        print(f"Saved {len(errors)}/{TRIALS} to {RESULTS} | "
              f"MAE {sum(map(abs, errors)) / len(errors):.3f} m | "
              f"signed error range {min(errors):+.3f} to {max(errors):+.3f} m")


def main():
    errors = saved_errors()
    if len(errors) >= TRIALS:
        show_summary(errors)
        return

    positions = deque(maxlen=1)

    def on_position(data):
        x, y = map(float, data[:2])
        if math.isfinite(x) and math.isfinite(y):
            positions.append((time.monotonic(), x, y))

    ep = robot.Robot()
    streaming = subscribed = False
    try:
        ep.initialize(conn_type="sta", proto_type="tcp")
        subscribed = bool(ep.chassis.sub_position(cs=1, freq=20, callback=on_position))
        streaming = bool(ep.camera.start_video_stream(display=False))
        if not (subscribed and streaming):
            raise RuntimeError("Could not start camera and chassis odometry.")

        while len(errors) < TRIALS:
            number = len(errors) + 1
            print(f"\nTrial {number}/{TRIALS}. Reset the camera lens to 1.50 m from the fixed tag.")
            if input("Aim at the tag, clear the path, and press Enter (or q): ").strip().lower() == "q":
                break
            try:
                result = run_trial(ep, positions)
            except RuntimeError as error:
                print(f"Trial incomplete: {error}", flush=True)
                continue

            print(f"Stopped. Estimated remaining distance: "
                  f"{result['estimated_remaining_m']:.3f} m.")
            measured = tape_reading()
            if measured is None:
                break
            errors.append(save_trial(number, result, measured))
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")
    finally:
        show_summary(errors)
        for active, close in (
            (subscribed, lambda: ep.chassis.unsub_position()),
            (streaming, lambda: ep.camera.stop_video_stream()),
            (True, ep.close),
        ):
            if active:
                try:
                    close()
                except Exception as error:
                    print(f"SDK cleanup: {error}", flush=True)


if __name__ == "__main__":
    main()
