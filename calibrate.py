"""Capture calibration images from the EP camera and fit camera intrinsics."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from phase3 import connect, detect, FAMILIES, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    marker = sub.add_parser("marker")
    marker.add_argument("--output", type=Path, required=True)
    marker.add_argument("--family", choices=["16h5", "25h9", "36h10", "36h11"], default="36h11")
    marker.add_argument("--id", type=int, default=0)
    capture = sub.add_parser("capture")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--connection", choices=["sta", "ap", "rndis"], default="sta")
    fit = sub.add_parser("fit")
    fit.add_argument("--images", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--cols", type=int, default=9)
    fit.add_argument("--rows", type=int, default=6)
    tagfit = sub.add_parser("fit-tag")
    tagfit.add_argument("--images", type=Path, required=True)
    tagfit.add_argument("--output", type=Path, required=True)
    tagfit.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    if args.mode == "marker":
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, "DICT_APRILTAG_" + args.family))
        if hasattr(cv2.aruco, "generateImageMarker"):
            tag = cv2.aruco.generateImageMarker(dictionary, args.id, 600)
        else:
            tag = cv2.aruco.drawMarker(dictionary, args.id, 600)
        tag = np.pad(tag, 100, constant_values=255)
        if not cv2.imwrite(str(args.output), tag):
            raise RuntimeError("Unable to write AprilTag target")
        return

    if args.mode == "capture":
        args.output.mkdir(parents=True, exist_ok=True)
        ep = connect(args.connection)
        camera = ep.camera
        try:
            if not camera.start_video_stream(display=False):
                raise RuntimeError("Camera stream failed")
            for number in range(1, 100):
                answer = input("Hold the tag in a new position and tilt. Enter to capture or q to finish: ")
                if answer.lower() == "q":
                    break
                frame = camera.read_cv2_image(timeout=2, strategy="newest")
                path = args.output / "board_{:02d}.png".format(number)
                if frame is None or not cv2.imwrite(str(path), frame):
                    raise RuntimeError("Unable to capture image")
                print(path)
        finally:
            try:
                camera.stop_video_stream()
            finally:
                ep.close()
        return

    if args.mode == "fit-tag":
        cfg = read_config(args.config)
        if cfg["tag_family"] not in FAMILIES:
            raise ValueError("Unsupported AprilTag family in config")
        half = cfg["tag_side_m"] / 2
        object_corners = np.array(
            [[-half, half, 0], [half, half, 0],
             [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, "DICT_APRILTAG_" + cfg["tag_family"]))
        points, pixels, size = [], [], None
        for path in sorted(args.images.glob("*.png")):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if size is None:
                size = frame.shape[1], frame.shape[0]
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError("All images must have the same resolution")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = detect(gray, cfg["tag_family"])
            matching = [] if ids is None else np.flatnonzero(ids.ravel() == cfg["tag_id"])
            if len(matching) == 1:
                points.append(object_corners.copy())
                pixels.append(corners[int(matching[0])].reshape(4, 2).astype(np.float32))
            else:
                print("Configured AprilTag not detected in", path)
        if len(points) < 12:
            raise ValueError("At least 12 sharp views of the configured tag are required")
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(points, pixels, size, None, None)
        print("{} AprilTag views, reprojection RMS {:.3f} px".format(len(points), rms))
        if not np.isfinite(rms) or rms > 2.0:
            raise ValueError("Collect sharper tag views with wider changes in tilt and image position")
        np.savez(args.output, K=matrix, dist=distortion, size=size, rms=rms,
                 tag_side_m=cfg["tag_side_m"])
        return

    pattern = (args.cols, args.rows)
    board = np.zeros((args.cols * args.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    points, pixels, size = [], [], None
    for path in sorted(args.images.glob("*.png")):
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        if size is None:
            size = frame.shape[::-1]
        if frame.shape[::-1] != size:
            raise ValueError("All images must have the same resolution")
        ok, corners = cv2.findChessboardCornersSB(frame, pattern)
        if ok:
            points.append(board)
            pixels.append(corners)
        else:
            print("Board not detected in", path)
    if len(points) < 10:
        raise ValueError("At least 10 detected checkerboards are required")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(points, pixels, size, None, None)
    print("{} views, reprojection RMS {:.3f} px".format(len(points), rms))
    if not np.isfinite(rms) or rms > 1.0:
        raise ValueError("Collect sharper views with more varied board positions and tilts")
    np.savez(args.output, K=matrix, dist=distortion, size=size, rms=rms)


if __name__ == "__main__":
    main()
