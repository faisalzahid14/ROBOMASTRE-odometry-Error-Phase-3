# ROBOMASTRE odometry Error Phase 3

RoboMaster EP Core odometry experiment for SCE 672 at KFUPM. The [report](RoboMaster_Phase3_Report.pdf) contains the method, formulas and theory, AI-assisted code assessment, eight trial scatter plots, a final scatter plot, error calculations, and statistical analysis. The [assignment brief](RoboMaster_Odometry_Project_Phase_3-2.pdf) specifies the submission requirements.

## Recorded experiment

`phase3_eight.py` is the script used for the eight recorded trials. It detects AprilTag 25h9 ID 3 with the onboard camera, commands fourteen 0.10 m straight forward steps from a 1.50 m start, predicts remaining distance from chassis odometry, and corrects it with camera apparent-width range. The independent tape distance is measured from the camera lens to the tag centre after each stop. The report explains that the assignment requested a QR tag, while this experiment used an AprilTag.

`phase3_eight_measurements_corrected.csv` is the dataset used in the report. The original `phase3_eight_measurements.csv` is retained for provenance. Its Trial 1 tape reading was entered as 1.70 m; the report documents its correction to 0.10 m. No other trial values were changed.

The remaining Python files are development, calibration, detection, capture, and analysis scripts used during the project. `config.json` applies to the earlier `phase3.py` workflow; `phase3_eight.py` uses constants defined in that script. The target images are included.

## Run

Use Ubuntu with a connected RoboMaster EP Core and the DJI RoboMaster Python SDK. Install the other Python dependencies with `pip install -r requirements.txt`. Mount the supplied AprilTag, set the camera lens 1.50 m from its centre, and run `python3 phase3_eight.py`. Follow the terminal prompts and measure from the camera lens to the tag centre after each trial. The script writes `phase3_eight_measurements.csv`; keep the original dataset safe before a new run.

## Submission status

The [report](RoboMaster_Phase3_Report.pdf), code, and [demonstration video](demonstration.mp4) are included. The [GitHub repository](https://github.com/faisalzahid14/ROBOMASTRE-odometry-Error-Phase-3) hosts the submission. Regenerate the PDF from the corrected CSV with `MPLCONFIGDIR=/tmp/mpl_phase3 python3 build_submission_report.py`. The graph for each trial compares its two recorded final values; the experiment did not log per-step estimates. The Trial 1 tape correction should be checked against field notes or video because the logged CSV alone cannot verify it.
