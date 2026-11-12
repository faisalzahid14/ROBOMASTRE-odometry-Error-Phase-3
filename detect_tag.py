"""Identify supported AprilTags in a saved camera image. No robot connection."""
import argparse
from pathlib import Path

FAMILIES = ("16h5", "25h9", "36h10", "36h11")


def detect(gray, family):
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_APRILTAG_" + family))
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)[:2]
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("camera_check.jpg"))
    args = parser.parse_args()
    print("OpenCV", cv2.__version__)
    if not hasattr(cv2, "aruco"):
        raise SystemExit("This OpenCV build lacks aruco. Send this output before changing your environment.")
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit("Cannot open " + str(args.image))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = False
    for family in FAMILIES:
        if not hasattr(cv2.aruco, "DICT_APRILTAG_" + family):
            continue
        corners, ids = detect(gray, family)
        if ids is not None:
            found = True
            print("Family tag{} | IDs {}".format(family, ids.ravel().tolist()))
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    if not found:
        print("No AprilTag decoded. Check tag family, focus, lighting, size and complete border.")
    else:
        output = args.image.with_name("apriltag_check.jpg")
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError("Could not save annotated image")
        print("Saved", output)


if __name__ == "__main__":
    main()
