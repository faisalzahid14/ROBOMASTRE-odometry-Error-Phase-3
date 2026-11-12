"""Build the SCE 672 Phase 3 PDF report from paired trial logs and measurements."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from phase3 import read_config


def results(measurements, runs):
    records = []
    if measurements is None:
        return records
    with measurements.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["trial"].strip()
            actual = float(row["measured_m"])
            if not name.isdigit() or not math.isfinite(actual) or actual <= 0:
                raise ValueError("Use positive distances in metres and numeric trial IDs")
            if any(record[0] == name for record in records):
                raise ValueError("Duplicate trial ID " + name)
            log = json.loads((runs / ("trial" + name) / "summary.json").read_text())
            if log["status"] != "stop_distance_reached":
                raise ValueError("Trial {} did not complete: {}".format(name, log["status"]))
            estimate, wheel = float(log["estimate_m"]), float(log["wheel_only_m"])
            if not all(math.isfinite(d) and d >= 0 for d in (estimate, wheel)):
                raise ValueError("Invalid estimate in trial " + name)
            records.append((name, estimate, actual, estimate - actual, wheel - actual, log))
    if not 8 <= len(records) <= 10:
        raise ValueError("The brief requires 8 to 10 completed trials")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-id", default="")
    parser.add_argument("--repository", default="")
    parser.add_argument("--video", default="")
    args = parser.parse_args()
    data = results(args.measurements, args.runs)
    cfg = read_config(args.config)
    if any(r[5]["configuration"] != cfg for r in data):
        raise ValueError("The trial settings differ from the selected configuration")

    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    pdfmetrics.registerFont(TTFont("Vera", str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", str(font_dir / "VeraBd.ttf")))
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Vera"
    styles["Title"].fontName = "Vera-Bold"
    styles["Heading2"].fontName = "Vera-Bold"
    styles.add(ParagraphStyle("Kicker", fontName="Vera-Bold", fontSize=9,
                              textColor=colors.HexColor("#29475B"), spaceAfter=9))
    styles["Title"].alignment = TA_CENTER
    styles["Title"].fontSize = 23
    styles["Title"].leading = 27
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 14
    styles["Normal"].spaceAfter = 7
    styles["Heading2"].spaceBefore = 11
    styles["Heading2"].spaceAfter = 5

    story = []
    def add(text, style="Normal"):
        story.append(Paragraph(text, styles[style]))

    add("KING FAHD UNIVERSITY OF PETROLEUM AND MINERALS", "Kicker")
    add("RoboMaster Odometry Phase 3", "Title")
    add("SCE 672 Advanced Robotics &nbsp; | &nbsp; Prof. Aamer Iqbal Bhatti")
    add("Faisal Zahid &nbsp; | &nbsp; Student ID " + escape(args.student_id or "____________"))
    if not data:
        add("WORKING DRAFT &nbsp; | &nbsp; Physical results pending", "Kicker")
    add("Objective", "Heading2")
    add("Estimate the RoboMaster EP Core's remaining distance to a fixed AprilTag tag during a "
        "straight approach. Compare the estimate at rest with an independent tape measurement.")
    add("Experimental Setup", "Heading2")
    add("The robot starts aligned with a fixed, vertical AprilTag tag on a level floor. The configured "
        "starting distance is {:.2f} m, the stopping threshold is {:.2f} m, and the "
        "forward command is {:.2f} m/s. The distance reference is the floor projection of "
        "the chassis center to the floor projection of the AprilTag center. The same floor "
        "marks, camera position, and thresholds apply to every trial.".format(
            cfg["start_m"], cfg["stop_m"], cfg["speed_mps"]))
    add("Prediction and Correction", "Heading2")
    add("The EKF state is [p<sub>x</sub>, p<sub>y</sub>, heading, v, yaw rate]. Its motion "
        "model holds the most recently estimated speed and yaw rate between events and propagates "
        "the pose. At each new ESC packet, Mecanum kinematics convert four wheel RPMs into "
        "forward speed and yaw rate for a wheel update. Direct integration of the same wheel "
        "rates supplies a wheel-only baseline.")
    add("The onboard camera detects the configured AprilTag family and ID. The four image corners, the "
        "measured AprilTag size, and the calibrated camera intrinsics give the AprilTag center in "
        "camera coordinates. After applying the camera mounting offset and pitch, the "
        "horizontal range and bearing correct the EKF using the surveyed tag location. "
        "The bearing innovation is wrapped. The controller commands straight motion and "
        "stops when the valid camera range reaches the configured threshold.")
    add("Covariance Settings", "Heading2")
    add("The prediction covariance increment is Q(dt) = diag(s²) × dt, with five state noise "
        "parameters s = {} per square root second. Wheel standard deviations are {} (m/s, rad/s). "
        "Tag standard deviations are {} (m, rad). Each measurement covariance is diagonal "
        "with the squared standard deviations.".format(cfg["process_sd_per_sqrt_s"],
        cfg["wheel_sd"], cfg["tag_sd"]))
    add("Error Calculation", "Heading2")
    add("Signed error = estimated final range minus measured final range. Positive values "
        "mean the EKF overestimated the remaining range. Mean absolute error (MAE) averages "
        "the absolute values of the signed errors. The range is reported as the smallest "
        "and largest signed errors, with the span also given.")
    story.append(PageBreak())
    add("Trial Results", "Heading2")

    rows = [["Trial", "EKF estimate (m)", "Tape (m)", "Signed error (m)", "Wheel error (m)"]]
    for trial, estimate, actual, err, wheel_err, _ in data:
        rows.append([trial, "{:.3f}".format(estimate), "{:.3f}".format(actual),
                     "{:+.3f}".format(err), "{:+.3f}".format(wheel_err)])
    if not data:
        rows.extend([[str(i), "", "", "", ""] for i in range(1, 9)])
    table = Table(rows, colWidths=[19*mm, 39*mm, 25*mm, 38*mm, 36*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#29475B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Vera-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Vera"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EFF4F7"), colors.white]),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#29475B")),
    ]))
    story.append(table)
    story.append(Spacer(1, 7))
    add("Discussion", "Heading2")
    if data:
        errors = [r[3] for r in data]
        wheel_errors = [r[4] for r in data]
        actuals = [r[2] for r in data]
        mae, wheel_mae = mean(map(abs, errors)), mean(map(abs, wheel_errors))
        add("Across {} trials, the EKF MAE was {:.3f} m and the wheel-only MAE was {:.3f} m. "
            "Signed EKF errors ranged from {:+.3f} to {:+.3f} m (span {:.3f} m). "
            "Measured stopping distances ranged from {:.3f} to {:.3f} m.".format(
                len(data), mae, wheel_mae, min(errors), max(errors), max(errors)-min(errors),
                min(actuals), max(actuals)))
        sign = "overestimated" if mean(errors) > 0 else "underestimated"
        if abs(mean(errors)) >= .0005:
            add("On average, the EKF {} the remaining distance by {:.3f} m.".format(sign, abs(mean(errors))))
        else:
            add("The average signed EKF error rounded to zero at millimetre precision.")
    else:
        add("Physical trials pending. Enter eight to ten independently measured distances to "
            "populate the table, compute MAE, and compare the error range with the wheel baseline.")
    add("Wheel slip and an inaccurate effective wheel radius can accumulate pose error in "
        "prediction. AprilTag print scale, camera calibration, mounting offsets, latency, and "
        "tape alignment can affect the correction or reference measurement. The camera "
        "provides a distance reference the wheel readings alone do not observe.")
    add("Submission Links", "Heading2")
    for label, url in (("GitHub repository", args.repository), ("Demonstration video", args.video)):
        add(label + " " + escape(url or "pending"))

    def footer(canvas, doc):
        canvas.setFont("Vera", 8)
        canvas.setFillColor(colors.HexColor("#29475B"))
        canvas.drawString(20*mm, 13*mm, "SCE 672   |   RoboMaster Odometry")
        canvas.drawRightString(doc.pagesize[0]-20*mm, 13*mm, str(doc.page))
    doc = SimpleDocTemplate(str(args.output), leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=17*mm, bottomMargin=18*mm,
                            title="RoboMaster Odometry Phase 3", author="Faisal Zahid")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
