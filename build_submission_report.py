"""Regenerate the submission PDF from the recorded eight-trial CSV."""
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'phase3_eight_measurements_corrected.csv'
OUTPUT = ROOT / 'RoboMaster_Phase3_Report.pdf'
REPO = 'https://github.com/faisalzahid14/ROBOMASTRE-odometry-Error-Phase-3'
VIDEO = REPO + '/blob/main/demonstration.mp4'

with DATA.open(newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 8
for row in rows:
    for key in ('estimated_remaining_m', 'measured_remaining_m', 'signed_error_m', 'odometry_travel_m'):
        row[key] = float(row[key])
    assert math.isclose(row['signed_error_m'], row['estimated_remaining_m'] - row['measured_remaining_m'], abs_tol=1e-9)

ids = [int(r['trial']) for r in rows]
est = [r['estimated_remaining_m'] for r in rows]
tape = [r['measured_remaining_m'] for r in rows]
err = [r['signed_error_m'] for r in rows]
mae = mean(map(abs, err))
rmse = math.sqrt(mean(e*e for e in err))
bias = mean(err)
wheel = [1.5-r['odometry_travel_m'] for r in rows]
wheel_mae = mean(abs(w-t) for w,t in zip(wheel,tape))

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9})
with PdfPages(OUTPUT) as pdf:
    fig = plt.figure(figsize=(8.27,11.69))
    fig.text(.08,.95,'RoboMaster Odometry - Phase 3',fontsize=19,weight='bold')
    fig.text(.08,.925,'SCE 672 | Faisal Zahid | KFUPM',fontsize=10)
    sections = [
        ('Submission evidence', f'GitHub: {REPO}\nVideo: {VIDEO}\nSource data: phase3_eight_measurements_corrected.csv; original CSV retained.'),
        ('Experiment and theory', 'Eight runs began at 1.50 m from a fixed AprilTag 25h9 ID 3. The robot commanded fourteen 0.10 m forward moves. Chassis position changes predicted remaining range; apparent AprilTag width supplied range corrections. Final tape distance was measured from camera lens to tag centre. The assignment specifies a QR tag; this run used an AprilTag.'),
        ('Prediction and correction formulas', 'Odometry increment: d = sqrt((x_k-x_{k-1})^2 + (y_k-y_{k-1})^2).\nPrediction: r_pred = r_(k-1) - d; P_pred = P_(k-1) + (0.025 m)^2.\nCamera range: z_k = 1.50 m * w_start / w_k (apparent-width ratio).\nCorrection: K_k = P_pred / (P_pred + (0.10 m)^2); r_k = r_pred + K_k(z_k-r_pred); P_k = (1-K_k)P_pred.\nA camera reading is accepted only when |z_k-r_pred| <= 0.30 m; r_k is clipped at zero. Invalid odometry adds (0.10 m)^2 to variance.'),
        ('Error calculation and correction', 'Signed error e_i = estimated final range - tape final range; absolute error = |e_i|. Positive error means overestimation. The corrected CSV changes Trial 1 tape entry from 1.70 m to 0.10 m; its estimate is 0.00 m, giving e_1 = -0.10 m. The original CSV remains in the repository. The basis for this manual correction cannot be verified from the logged measurements; check it against the video or field notes before submission.'),
        ('AI-assisted code assessment', 'The recorded script checks the tag ID, requires fresh chassis position, rejects implausible step changes, gates camera updates, and saves all eight final estimates and tape readings. Limitations: it uses planar odometry path length as forward progress; width ratio assumes a fixed tag angle and scale; it clips negative estimates to zero; fixed move commands do not enforce a fixed final clearance. No per-step estimates are saved, so the plots on later pages show final results only. This is a code review, not an independently verified physical test.')
    ]
    y=.875
    for title,body in sections:
        fig.text(.08,y,title,fontsize=11,weight='bold',va='top')
        import textwrap
        lines=[]
        for para in body.split('\n'):
            lines.extend(textwrap.wrap(para, 105, break_long_words=False, break_on_hyphens=False) or [''])
        y-=.022
        fig.text(.08,y,'\n'.join(lines),fontsize=8.5,va='top',linespacing=1.5)
        y-=.023*len(lines)+.035
    pdf.savefig(fig); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.27,11.69)); ax.axis('off')
    fig.suptitle('Trial results and statistical analysis',fontsize=17,weight='bold',y=.96)
    cells=[[r['trial'],f'{a:.3f}',f'{b:.3f}',f'{e:+.3f}',f'{abs(e):.3f}'] for r,a,b,e in zip(rows,est,tape,err)]
    tab=ax.table(cellText=cells,colLabels=['Trial','Estimate (m)','Tape (m)','Error (m)','|Error| (m)'],loc='upper center',cellLoc='center',bbox=[.04,.53,.92,.32])
    tab.auto_set_font_size(False);tab.set_fontsize(9)
    summary=(f'n = 8     Mean signed error = {bias:+.3f} m     MAE = {mae:.3f} m\n'
             f'RMSE = {rmse:.3f} m     Sample SD of error = {stdev(err):.3f} m\n'
             f'Min / max signed error = {min(err):+.3f} / {max(err):+.3f} m     Span = {max(err)-min(err):.3f} m\n'
             f'Mean tape clearance = {mean(tape):.3f} m     Odometry-only MAE = {wheel_mae:.3f} m')
    fig.text(.09,.47,summary,fontsize=10,va='top',linespacing=1.8)
    fig.text(.09,.32,'Interpretation',fontsize=11,weight='bold')
    fig.text(.09,.29,'All eight estimates are below their tape readings. Camera correction reduced the\n'
             'descriptive MAE versus the odometry-only endpoint calculation, but this small\n'
             'sample does not establish performance on other surfaces or target poses. The\n'
             'final tape clearances vary from 0.10 to 0.50 m under the same commands.',fontsize=9,va='top',linespacing=1.5)
    fig.text(.09,.14,'Statistics use the corrected CSV. SD uses n-1; RMSE = sqrt(mean(e^2));\nMAE = mean(|e|). Each trial contributes one final observation.',fontsize=8.5)
    pdf.savefig(fig);plt.close(fig)

    fig,axes=plt.subplots(4,2,figsize=(8.27,11.69),sharey=True)
    fig.suptitle('Each trial - final range scatter plots',fontsize=16,weight='bold')
    for ax,i,a,b,e in zip(axes.flat,ids,est,tape,err):
        ax.scatter([0,1],[a,b],s=50,c=['#1768a6','#d55e00'])
        ax.set_xticks([0,1],['Estimate','Tape'])
        ax.set_ylim(-.04,.58);ax.grid(axis='y',alpha=.25)
        ax.set_title(f'Trial {i}: error {e:+.3f} m',fontsize=10)
        ax.set_ylabel('Remaining range (m)')
    fig.text(.5,.02,'Each panel contains the two recorded final values; no step-by-step series was logged.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]);pdf.savefig(fig);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.27,11.69))
    ax.scatter(ids,est,label='Estimated',color='#1768a6',s=55)
    ax.scatter(ids,tape,label='Tape measured',color='#d55e00',s=55,marker='s')
    for i,a,b in zip(ids,est,tape):ax.plot([i,i],[a,b],color='gray',alpha=.5)
    ax.set_xticks(ids);ax.set_xlabel('Trial');ax.set_ylabel('Final remaining range (m)')
    ax.set_title('Final comparison - scatter plot',fontsize=16,weight='bold')
    ax.grid(alpha=.25);ax.legend()
    fig.text(.5,.08,'Vertical separation equals the absolute final distance error for each trial.',ha='center',fontsize=9)
    pdf.savefig(fig);plt.close(fig)
print(f'Wrote {OUTPUT}; MAE={mae:.6f}, bias={bias:.6f}, wheel MAE={wheel_mae:.6f}')
