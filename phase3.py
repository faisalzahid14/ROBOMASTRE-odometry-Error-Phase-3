"""RoboMaster EP Core Phase 3. Live hardware access is isolated in live()."""
import argparse
import csv
import json
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import numpy as np

from detect_tag import detect, FAMILIES
H_WHEEL = np.eye(5)[3:]
RPM_PROJECTION = np.array([[1, -1, -1, 1], [-1, -1, -1, -1]]) / 4
FIELDS = ["time_s", "event", "z1", "z2", "x_m", "y_m", "heading_rad",
          "v_mps", "omega_rad_s", "estimate_m", "wheel_only_m",
          "innovation_1", "innovation_2", "accepted"]


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Filter:
    def __init__(self, config, q_scale=1., r_scale=1., wheel_r_scale=1.):
        self.x = np.zeros(5)
        self.P = np.diag(np.square(config["initial_sd"]))
        self.Q_rate = np.diag(np.square(config["process_sd_per_sqrt_s"])) * q_scale
        self.R_wheel = np.diag(np.square(config["wheel_sd"])) * wheel_r_scale
        self.R_tag = np.diag(np.square(config["tag_sd"])) * r_scale
        self.marker = np.array([config["start_m"], 0.])
        self.wheel_pose = np.zeros(3)
        self.last_wheel_time = None
        self.last_wheel = None
        self.t = 0.

    def predict(self, t):
        dt = t - self.t
        if dt < -1e-8:
            raise ValueError("Events must be in time order")
        _, _, heading, speed, rate = self.x
        c, s = math.cos(heading), math.sin(heading)
        F = np.eye(5)
        F[0, 2], F[0, 3] = -speed * s * dt, c * dt
        F[1, 2], F[1, 3] = speed * c * dt, s * dt
        F[2, 4] = dt
        self.x[:3] += dt * np.array([speed * c, speed * s, rate])
        self.x[2] = wrap(self.x[2])
        self.P = F @ self.P @ F.T + self.Q_rate * dt
        self.t = t

    def correction(self, innovation, H, R, gate=None):
        S = H @ self.P @ H.T + R
        if gate is not None and innovation @ np.linalg.solve(S, innovation) > gate:
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ innovation
        self.x[2] = wrap(self.x[2])
        A = np.eye(5) - K @ H
        covariance = A @ self.P @ A.T + K @ R @ K.T
        self.P = (covariance + covariance.T) / 2
        return True

    def observe(self, t, event, z):
        self.predict(t)
        if event == "wheel":
            if self.last_wheel_time is not None:
                dt = t - self.last_wheel_time
                speed, rate = (z + self.last_wheel) / 2
                mid = self.wheel_pose[2] + rate * dt / 2
                self.wheel_pose += dt * np.array([speed * math.cos(mid),
                                                 speed * math.sin(mid), rate])
            self.last_wheel_time, self.last_wheel = t, z
            e = z - H_WHEEL @ self.x
            return e, self.correction(e, H_WHEEL, self.R_wheel)
        dx, dy = self.marker - self.x[:2]
        q = dx * dx + dy * dy
        if q < 1e-8:
            return np.full(2, np.nan), False
        d = math.sqrt(q)
        predicted = np.array([d, wrap(math.atan2(dy, dx) - self.x[2])])
        H = np.array([[-dx/d, -dy/d, 0, 0, 0], [dy/q, -dx/q, -1, 0, 0]])
        e = z - predicted
        e[1] = wrap(e[1])
        return e, self.correction(e, H, self.R_tag, gate=9.21)

    def distance(self):
        return float(np.linalg.norm(self.marker - self.x[:2]))

    def row(self, t, event, z, innovation, accepted):
        return [t, event, *z, *self.x, self.distance(),
                np.linalg.norm(self.marker - self.wheel_pose[:2]), *innovation, int(accepted)]


def read_config(path):
    cfg = json.loads(path.read_text())
    if cfg.get("tag_family") not in FAMILIES:
        raise ValueError("Set tag_family to the detected AprilTag family")
    if not isinstance(cfg.get("tag_id"), int) or cfg["tag_id"] < 0:
        raise ValueError("Set tag_id to the detected AprilTag ID")
    limits = {"start_m": (1., 2.), "stop_m": (.30, .85), "speed_mps": (.03, .12),
              "tag_side_m": (.07, .35), "wheel_radius_m": (.02, .10),
              "half_length_plus_width_m": (.10, .60), "camera_forward_m": (-.5, .5),
              "camera_left_m": (-.5, .5), "camera_pitch_deg": (-30., 30.)}
    for name, (low, high) in limits.items():
        if not low <= cfg[name] <= high:
            raise ValueError("{} must be in [{}, {}]".format(name, low, high))
    if cfg["start_m"] - cfg["stop_m"] < .25 or cfg["stop_m"] <= cfg["camera_forward_m"] + .12:
        raise ValueError("Increase the clearance between the tag and camera at the stopping point")
    if cfg["forward_sign"] not in (-1, 1) or cfg["yaw_sign"] not in (-1, 1):
        raise ValueError("Wheel signs must be +1 or -1")
    for name, length in (("initial_sd", 5), ("process_sd_per_sqrt_s", 5),
                         ("wheel_sd", 2), ("tag_sd", 2)):
        values = np.asarray(cfg[name], dtype=float)
        if values.shape != (length,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Invalid covariance settings for " + name)
    return cfg


class AprilTag:
    def __init__(self, calibration, cfg):
        import cv2
        self.cv = cv2
        data = np.load(calibration)
        self.K, self.dist = data["K"], data["dist"]
        self.size = tuple(data["size"])
        self.cfg = cfg
        s = cfg["tag_side_m"] / 2
        self.square = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        self.family = cfg["tag_family"]
        self.tag_id = cfg["tag_id"]
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV build requires aruco for AprilTag detection")

    def observe(self, frame):
        if tuple(frame.shape[1::-1]) != self.size:
            raise ValueError("Camera frame size differs from calibration")
        gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids = detect(gray, self.family)
        if ids is None:
            return None
        matching = np.flatnonzero(ids.ravel() == self.tag_id)
        if len(matching) != 1:
            return None
        pixels = corners[int(matching[0])].reshape(4, 2).astype(float)
        if min(np.linalg.norm(np.roll(pixels, -1, axis=0) - pixels, axis=1)) < 24:
            return None
        ok, rotation, translation = self.cv.solvePnP(
            self.square, pixels, self.K, self.dist, flags=self.cv.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            return None
        projected, _ = self.cv.projectPoints(self.square, rotation, translation, self.K, self.dist)
        if np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - pixels)**2, axis=1))) > 3.:
            return None
        right, down, optical_depth = translation.ravel()
        if not .10 < optical_depth < 3.:
            return None
        pitch = math.radians(self.cfg["camera_pitch_deg"])
        forward = (self.cfg["camera_forward_m"] + optical_depth * math.cos(pitch)
                   + down * math.sin(pitch))
        left = self.cfg["camera_left_m"] - right
        if forward <= 0:
            return None
        return np.array([math.hypot(forward, left), math.atan2(left, forward)])


class AprilTagLock:
    """Detect the configured tag and return its horizontal image offset."""
    def __init__(self, cfg):
        self.cfg = cfg

    def observe(self, frame):
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids = detect(gray, self.cfg["tag_family"])
        if ids is None:
            return None
        matching = np.flatnonzero(ids.ravel() == self.cfg["tag_id"])
        if len(matching) != 1:
            return None
        points = corners[int(matching[0])].reshape(4, 2)
        return 2.0 * (float(points[:, 0].mean()) / frame.shape[1] - 0.5)


def connect(connection):
    from robomaster import robot
    ep = robot.Robot()
    try:
        ep.initialize(conn_type=connection, proto_type="tcp")
    except BaseException:
        ep.close()
        raise
    return ep


def live(args):
    cfg = read_config(args.config)
    calibrated_camera = args.calibration is not None
    qr = AprilTag(args.calibration, cfg) if calibrated_camera else AprilTagLock(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    filter_ = Filter(cfg)
    events = SimpleQueue()
    scale = cfg["wheel_radius_m"] * 2 * np.pi / 60
    scales = scale * np.array([cfg["forward_sign"],
                              cfg["yaw_sign"] / cfg["half_length_plus_width_m"]])
    ep = connect(args.connection)
    chassis, camera = ep.chassis, ep.camera
    subscribed = streaming = driving = False
    last_camera = last_wheel = float("-inf")
    last_tag_sample = float("-inf")
    last_command = float("-inf")
    stop_at = None
    stable_since = None
    centered = 0
    reason = "initialization_failed"
    final_camera = None
    final_camera_accepted = False
    last_z = np.array([np.inf, np.inf])
    t0 = time.monotonic()

    def on_esc(data):
        rpm = np.asarray(data[0], dtype=float)
        if rpm.shape == (4,) and np.isfinite(rpm).all():
            events.put((time.monotonic() - t0, (RPM_PROJECTION @ rpm) * scales))

    with (args.output / "events.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)

        def wheel_updates():
            nonlocal last_wheel, last_z
            while True:
                try:
                    ts, z = events.get_nowait()
                except Empty:
                    break
                ts = max(ts, filter_.t)
                innovation, accepted = filter_.observe(ts, "wheel", z)
                writer.writerow(filter_.row(ts, "wheel", z, innovation, accepted))
                last_wheel, last_z = ts, z

        try:
            subscribed = bool(chassis.sub_esc(freq=20, callback=on_esc))
            if not subscribed:
                raise RuntimeError("ESC subscription failed")
            streaming = bool(camera.start_video_stream(display=False))
            if not streaming:
                raise RuntimeError("Camera stream failed")
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                try:
                    frame = camera.read_cv2_image(timeout=.15, strategy="newest")
                except Empty:
                    frame = None
                frame_time = time.monotonic() - t0
                detection = qr.observe(frame) if frame is not None else None
                wheel_updates()
                now = time.monotonic() - t0
                if detection is not None and frame_time - last_tag_sample >= .15:
                    ts = max(frame_time, filter_.t)
                    last_tag_sample = frame_time
                    final_camera = detection
                    if args.probe:
                        if calibrated_camera:
                            print("Camera range {:.3f} m | bearing {:+.2f} deg".format(
                                detection[0], math.degrees(detection[1])))
                        else:
                            print("AprilTag detected | horizontal image offset {:+.3f}".format(detection))
                        reason = "probe_ok"
                        break
                    if calibrated_camera:
                        e, accepted = filter_.observe(ts, "tag", detection)
                        writer.writerow(filter_.row(ts, "tag", detection, e, accepted))
                        final_camera_accepted = accepted
                        if accepted:
                            last_camera = frame_time
                        tag_centered = accepted and abs(detection[1]) <= math.radians(6)
                    else:
                        last_camera = frame_time
                        tag_centered = abs(detection) <= .12
                    if not driving and stop_at is None:
                        centered = centered + 1 if (
                            tag_centered and now - last_wheel < .25 and abs(last_z[0]) < .025
                            and abs(last_z[1]) < .08) else 0
                if not driving and stop_at is None:
                    if centered >= 3 and now - last_camera < .35:
                        if not args.drive:
                            reason = "preflight_ok"
                            break
                        driving = True
                        print("Target locked. Straight approach started.", flush=True)
                    elif now > 10.:
                        reason = "preflight_failed"
                        break
                if driving and stop_at is None:
                    if now - last_camera > .40 or now - last_wheel > .25:
                        reason = "sensor_timeout"
                    elif final_camera is not None and (
                        abs(final_camera[1]) > math.radians(12) if calibrated_camera
                        else abs(final_camera) > .24):
                        reason = "target_off_axis"
                    elif calibrated_camera and final_camera is not None and final_camera[0] <= cfg["stop_m"]:
                        reason = "stop_distance_reached" if final_camera_accepted else "unconfirmed_close_reading"
                    elif not calibrated_camera and cfg["start_m"] - filter_.wheel_pose[0] <= cfg["stop_m"]:
                        reason = "stop_distance_reached"
                    elif filter_.wheel_pose[0] >= cfg["start_m"] - cfg["stop_m"] + .15:
                        reason = "wheel_travel_limit"
                    else:
                        if now - last_command >= .10:
                            if chassis.drive_speed(x=cfg["speed_mps"], y=0, z=0, timeout=.30) is False:
                                raise RuntimeError("Forward command failed")
                            last_command = now
                        continue
                    chassis.drive_speed(x=0, y=0, z=0, timeout=.30)
                    stop_at = now
                if stop_at is not None:
                    if reason != "stop_distance_reached":
                        break
                    stationary = now - last_wheel < .25 and abs(last_z[0]) < .025 and abs(last_z[1]) < .08
                    stable_since = (now if stable_since is None else stable_since) if stationary else None
                    if (stable_since is not None and now - stable_since >= .5
                            and now - last_camera < .4 and last_camera > stable_since):
                        break
                    if now - stop_at > 2.:
                        reason = "final_pose_unconfirmed"
                        break
            else:
                reason = "time_limit"
        except KeyboardInterrupt:
            reason = "operator_stop"
        except Exception:
            reason = "runtime_error"
            raise
        finally:
            try:
                if args.drive:
                    chassis.drive_speed(x=0, y=0, z=0, timeout=.30)
            finally:
                try:
                    try:
                        if streaming:
                            camera.stop_video_stream()
                    finally:
                        if subscribed:
                            chassis.unsub_esc()
                finally:
                    ep.close()
                    if calibrated_camera:
                        estimate = filter_.distance()
                        wheel_estimate = float(np.linalg.norm(filter_.marker - filter_.wheel_pose[:2]))
                    else:
                        estimate = cfg["start_m"] - filter_.x[0]
                        wheel_estimate = cfg["start_m"] - filter_.wheel_pose[0]
                    summary = {"status": reason, "estimate_m": float(estimate),
                               "wheel_only_m": float(wheel_estimate),
                               "camera_mode": "range_bearing" if calibrated_camera else "tag_lock_only",
                               "start_m": cfg["start_m"], "stop_m": cfg["stop_m"],
                               "configuration": cfg}
                    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                    print("{} | EKF final range {:.3f} m".format(reason, summary["estimate_m"]))
    if reason not in ("stop_distance_reached", "preflight_ok", "probe_ok"):
        raise SystemExit(1)


def replay(args):
    cfg = read_config(args.config)
    result = Filter(cfg, args.q_scale, args.r_tag_scale, args.r_wheel_scale)
    with args.input.open(newline="") as source, args.output.open("x", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(FIELDS)
        for row in csv.DictReader(source):
            t = float(row["time_s"])
            z = np.array([float(row["z1"]), float(row["z2"])])
            e, accepted = result.observe(t, row["event"], z)
            writer.writerow(result.row(t, row["event"], z, e, accepted))
    print("Final estimated range {:.3f} m".format(result.distance()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("live")
    motion = run.add_mutually_exclusive_group()
    motion.add_argument("--drive", action="store_true", help="Enable forward movement")
    motion.add_argument("--probe", action="store_true", help="Print the first valid camera range without motion")
    run.add_argument("--connection", choices=["ap", "sta", "rndis"], default="sta")
    run.add_argument("--calibration", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path, default=Path("config.json"))
    offline = sub.add_parser("replay")
    offline.add_argument("--input", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--config", type=Path, default=Path("config.json"))
    for option in ("q-scale", "r-wheel-scale", "r-tag-scale"):
        offline.add_argument("--" + option, type=float, default=1.)
    args = parser.parse_args()
    if args.mode == "replay":
        if any(not np.isfinite(v) or v <= 0 for v in (args.q_scale, args.r_wheel_scale, args.r_tag_scale)):
            parser.error("Noise scales must be positive")
        replay(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
