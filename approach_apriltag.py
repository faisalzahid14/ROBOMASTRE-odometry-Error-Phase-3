import time
from queue import Empty

import cv2
import numpy as np
from robomaster import robot

from detect_tag import detect

TAG_FAMILY = "25h9"
TAG_ID = 3
STEP_M = 0.10
MAX_STEPS = 12
MOVE_TIMEOUT_S = 8
STOP_WIDTH = 0.08  # Stop when the tag spans 8% of the image width
CENTRE_TOLERANCE = 0.10  # Accept the tag centre between 40% and 60%


def find_tag(camera):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            frame = camera.read_cv2_image(timeout=0.25, strategy="newest")
        except (Empty, TimeoutError):
            continue
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(gray, TAG_FAMILY)
        if ids is None:
            continue

        matches = np.flatnonzero(ids.ravel() == TAG_ID)
        if len(matches) == 1:
            x = corners[int(matches[0])].reshape(4, 2)[:, 0] / frame.shape[1]
            return float(x.mean()), float(np.ptp(x))

    return None


def main():
    ep = robot.Robot()
    ep.initialize(conn_type="sta", proto_type="tcp")
    streaming = False

    try:
        streaming = bool(ep.camera.start_video_stream(display=False))
        if not streaming:
            raise RuntimeError("Camera stream did not start.")

        completed_steps = 0
        while True:
            tag = find_tag(ep.camera)
            if tag is None:
                print("Stopped: AprilTag not visible.")
                return

            centre, width = tag
            print(f"Tag centre {centre:.1%} | width {width:.1%}", flush=True)

            if width >= STOP_WIDTH:
                print("Stopped: tag size threshold reached.")
                return
            if abs(centre - 0.5) > CENTRE_TOLERANCE:
                print("Stopped: tag off centre. Realign the robot before restarting.")
                return
            if completed_steps >= MAX_STEPS:
                print("Stopped: 1.2 m commanded travel limit reached.")
                return

            step = completed_steps + 1
            print(f"Step {step}: moving forward {STEP_M:.2f} m.", flush=True)
            action = ep.chassis.move(x=STEP_M, y=0, z=0, xy_speed=0.5)

            if not action.wait_for_completed(timeout=MOVE_TIMEOUT_S):
                print(f"Stopped issuing moves: step {step} timed out. {action}")
                return
            if getattr(action, "has_succeeded", True) is False:
                print(f"Stopped: step {step} failed. {action}")
                return

            completed_steps += 1
            time.sleep(0.4)
    finally:
        try:
            if streaming:
                ep.camera.stop_video_stream()
        finally:
            ep.close()


if __name__ == "__main__":
    main()
