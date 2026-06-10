#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# =============================================================================
# Single-command pipeline that executes every piece of the catch_robot
# project end-to-end:
#
#   1. Sanity checks       (module unit tests, vision accuracy)
#   2. LSTM training       (data collection + training, optional)
#   3. Experiments A–F     (with the --full grids)
#   4. Demo video + frames (overview .mp4 + extracted PNGs)
#   5. Diagnostic plots    (stereo accuracy diagnostic if available)
#
# All output (plots, CSVs, video, JSON summary) is collected under
# `results/<timestamp>/` so a single tarball captures the entire run.
#
# USAGE
# -----
#   bash run_all.sh                  # full pipeline (~5 min with PyTorch,
#                                    #                ~3 min without)
#   bash run_all.sh --quick          # skip --full and skip LSTM training
#                                    # (~1 min, useful for smoke tests)
#   bash run_all.sh --skip-train     # full sweeps, no LSTM training
#   bash run_all.sh --skip-vision    # also skip experiment E (real render)
#   bash run_all.sh --help           # show this help message
#
# The script is idempotent: re-running it creates a new timestamped folder
# under results/ without touching previous runs.
# =============================================================================

set -euo pipefail

# ---- locate project root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults / flags -------------------------------------------------------
QUICK=0
SKIP_TRAIN=0
SKIP_VISION=0

for arg in "$@"; do
    case "$arg" in
        --quick)        QUICK=1; SKIP_TRAIN=1; SKIP_VISION=1 ;;
        --skip-train)   SKIP_TRAIN=1 ;;
        --skip-vision)  SKIP_VISION=1 ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Try: bash run_all.sh --help"
            exit 1
            ;;
    esac
done

# ---- environment ------------------------------------------------------------
export MUJOCO_GL="${MUJOCO_GL:-egl}"     # respect existing setting; default egl
PYTHON="${PYTHON:-python}"

# ---- output folder ----------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/${STAMP}"
LOG="${OUT}/run_all.log"
mkdir -p "$OUT"

# ---- logging helpers --------------------------------------------------------
BOLD="$(printf '\033[1m')"
GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"
RED="$(printf '\033[31m')"
RESET="$(printf '\033[0m')"

banner() {
    local text="$1"
    printf "\n${BOLD}============================================================${RESET}\n"
    printf "${BOLD}  %s${RESET}\n" "$text"
    printf "${BOLD}============================================================${RESET}\n"
    echo "" >> "$LOG"
    echo "============================================================" >> "$LOG"
    echo "  $text" >> "$LOG"
    echo "============================================================" >> "$LOG"
}

step() {
    printf "${BOLD}[step]${RESET} %s\n" "$1"
    echo "[step] $1" >> "$LOG"
}

ok()   { printf "${GREEN}  ✓${RESET} %s\n" "$1"; echo "  ok: $1"   >> "$LOG"; }
warn() { printf "${YELLOW}  ⚠${RESET} %s\n" "$1"; echo "  warn: $1" >> "$LOG"; }
fail() { printf "${RED}  ✗${RESET} %s\n" "$1"; echo "  fail: $1" >> "$LOG"; }

T0=$(date +%s)
elapsed() {
    local t1=$(date +%s)
    printf '%dm%02ds' $(((t1 - T0) / 60)) $(((t1 - T0) % 60))
}

# Run a command, send stdout/stderr to log, surface only key lines.
runlog() {
    local title="$1"; shift
    step "$title"
    {
        echo ""
        echo "----- $title -----"
        echo "+ $*"
    } >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        ok "$title  (elapsed: $(elapsed))"
        return 0
    else
        fail "$title  — see $LOG"
        return 1
    fi
}

# =============================================================================
banner "catch_robot — all-in-one pipeline"
echo "  output     : $OUT"
echo "  log        : $LOG"
echo "  python     : $($PYTHON --version 2>&1)"
echo "  MUJOCO_GL  : $MUJOCO_GL"
echo "  quick      : $QUICK"
echo "  skip_train : $SKIP_TRAIN"
echo "  skip_vision: $SKIP_VISION"

# -----------------------------------------------------------------------------
# 0. Sanity — module unit tests + scene loads
# -----------------------------------------------------------------------------
banner "0. Sanity checks"

runlog "Module unit tests (test_modules.py)" \
    $PYTHON scripts/test_modules.py

runlog "Vision pipeline accuracy vs ground truth (smoke_test_vision.py)" \
    $PYTHON scripts/smoke_test_vision.py

# -----------------------------------------------------------------------------
# 1. LSTM training (optional)
# -----------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    banner "1. LSTM predictor training"

    if ! $PYTHON -c "import torch" >/dev/null 2>&1; then
        warn "PyTorch not installed — skipping LSTM training."
        warn "  To enable: pip install torch  (or pass --skip-train to silence this)"
    else
        # Data collection
        runlog "Collect 2000 simulated throws for training" \
            $PYTHON scripts/collect_data.py --n-throws 2000

        # Training
        EPOCHS=${EPOCHS:-30}
        runlog "Train LSTM predictor ($EPOCHS epochs)" \
            $PYTHON scripts/train_predictor.py --epochs "$EPOCHS"

        # Stash artifacts in the results folder
        if [[ -f data/lstm_predictor.pt ]]; then
            cp data/lstm_predictor.pt "$OUT/lstm_predictor.pt"
            ok "Saved checkpoint  →  $OUT/lstm_predictor.pt"
        fi
        if [[ -f data/training_curve.png ]]; then
            cp data/training_curve.png "$OUT/training_curve.png"
            ok "Saved training curve  →  $OUT/training_curve.png"
        fi
    fi
else
    banner "1. LSTM predictor training  [SKIPPED]"
fi

# -----------------------------------------------------------------------------
# 2. Experiments A–F (and E if not skipped)
# -----------------------------------------------------------------------------
banner "2. Experiments"

# Choose which experiments to run, and with --full or not.
if [[ "$QUICK" -eq 1 ]]; then
    EXP_LIST="ABCF"      # quick = no real-vision E, no LSTM D
    FULL_FLAG=""
else
    if [[ -f data/lstm_predictor.pt ]]; then
        EXP_LIST="ABCDF"
    else
        EXP_LIST="ABCF"
    fi
    FULL_FLAG="--full"
fi

if [[ "$SKIP_VISION" -eq 0 && "$QUICK" -eq 0 ]]; then
    EXP_LIST="${EXP_LIST}E"
fi

step "Running experiments: $EXP_LIST   (full=${FULL_FLAG:-no})"

# Run each experiment INDIVIDUALLY so the log clearly separates them.
EXP_BASE="data/experiments"
mkdir -p "$EXP_BASE"

for letter in $(echo "$EXP_LIST" | grep -o .); do
    case "$letter" in
        A) desc="A. throw-velocity sweep" ;;
        B) desc="B. vision-noise robustness" ;;
        C) desc="C. intercept-decision timeline" ;;
        D) desc="D. predictor ablation (analytical vs LSTM)" ;;
        E) desc="E. real-vision spot check" ;;
        F) desc="F. failure-mode breakdown" ;;
    esac
    runlog "Experiment $desc" \
        $PYTHON scripts/run_experiments.py --exp "$letter" $FULL_FLAG
done

# Collect all experiment artifacts. Each run_experiments.py invocation
# creates its own timestamped folder; merge them into $OUT/experiments/.
mkdir -p "$OUT/experiments"
shopt -s nullglob
for d in "$EXP_BASE"/*; do
    if [[ -d "$d" ]]; then
        name="$(basename "$d")"
        # Only copy folders created during this run (filter by timestamp prefix)
        if [[ "$name" > "$STAMP" || "$name" == "${STAMP}"* ]] 2>/dev/null; then
            cp -r "$d" "$OUT/experiments/$name"
        fi
    fi
done

# Flatten the plots into a single folder for easy access.
mkdir -p "$OUT/plots"
find "$OUT/experiments" -name "*.png" -exec cp {} "$OUT/plots/" \;
shopt -u nullglob

ok "Experiment plots collected  →  $OUT/plots/"

# -----------------------------------------------------------------------------
# 3. Diagnostic stereo plots (if the diagnose script exists)
# -----------------------------------------------------------------------------
if [[ -f scripts/diagnose_vision.py ]]; then
    banner "3. Stereo diagnostic plots"
    runlog "Run diagnose_vision.py" \
        $PYTHON scripts/diagnose_vision.py
    for f in /tmp/diagnose_vision.png /tmp/midflight_overlay.png \
             /tmp/stereo_pair.png; do
        if [[ -f "$f" ]]; then
            cp "$f" "$OUT/plots/$(basename "$f")"
            ok "Captured $(basename "$f")"
        fi
    done
fi

# -----------------------------------------------------------------------------
# 4. Demo video + extracted frames
# -----------------------------------------------------------------------------
banner "4. Demo video & frame extraction"

DEMO_OUT="$OUT/demo"
mkdir -p "$DEMO_OUT"

runlog "Render demo video (demo.py --use-gt)" \
    $PYTHON scripts/demo.py --use-gt --out-dir "$DEMO_OUT"

# The demo script writes demo_overview.mp4 and demo_stats.png to DEMO_OUT.
if [[ -f "$DEMO_OUT/demo_overview.mp4" ]]; then
    ok "Demo video      →  $DEMO_OUT/demo_overview.mp4"
else
    warn "Demo video not produced (check log)"
fi
if [[ -f "$DEMO_OUT/demo_stats.png" ]]; then
    ok "Demo stats plot →  $DEMO_OUT/demo_stats.png"
fi

# Extract individual frames from the video.  Uses ffmpeg if available
# (best quality), falls back to OpenCV via Python if not.
FRAMES_DIR="$DEMO_OUT/frames"
mkdir -p "$FRAMES_DIR"

if [[ -f "$DEMO_OUT/demo_overview.mp4" ]]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        step "Extract frames with ffmpeg (every 5th frame)"
        if ffmpeg -y -i "$DEMO_OUT/demo_overview.mp4" \
                  -vf "select=not(mod(n\,5))" -vsync vfr \
                  "$FRAMES_DIR/frame_%03d.png" \
                  >> "$LOG" 2>&1; then
            N=$(ls "$FRAMES_DIR"/frame_*.png 2>/dev/null | wc -l)
            ok "Extracted $N frames  →  $FRAMES_DIR/"
        else
            warn "ffmpeg failed — trying Python fallback"
            $PYTHON - <<PYEOF >> "$LOG" 2>&1
import cv2
v = cv2.VideoCapture("$DEMO_OUT/demo_overview.mp4")
n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
idx = 0
saved = 0
while True:
    ok_, frame = v.read()
    if not ok_: break
    if idx % 5 == 0:
        cv2.imwrite(f"$FRAMES_DIR/frame_{saved:03d}.png", frame)
        saved += 1
    idx += 1
v.release()
print(f"Saved {saved} frames")
PYEOF
            ok "Frames extracted with OpenCV fallback"
        fi
    else
        step "ffmpeg not found — extracting frames with OpenCV"
        $PYTHON - <<PYEOF >> "$LOG" 2>&1
import cv2
v = cv2.VideoCapture("$DEMO_OUT/demo_overview.mp4")
n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
idx = 0
saved = 0
while True:
    ok_, frame = v.read()
    if not ok_: break
    if idx % 5 == 0:
        cv2.imwrite(f"$FRAMES_DIR/frame_{saved:03d}.png", frame)
        saved += 1
    idx += 1
v.release()
print(f"Saved {saved} of {n} frames")
PYEOF
        N=$(ls "$FRAMES_DIR"/frame_*.png 2>/dev/null | wc -l)
        ok "Extracted $N frames with OpenCV  →  $FRAMES_DIR/"
    fi

    # Build a 5-frame strip suitable for the presentation slide.
    step "Build 5-frame summary strip"
    $PYTHON - <<PYEOF >> "$LOG" 2>&1
import cv2, numpy as np, glob, os
frames = sorted(glob.glob("$FRAMES_DIR/frame_*.png"))
if len(frames) >= 5:
    picks = [frames[int(i * (len(frames)-1) / 4)] for i in range(5)]
else:
    picks = frames
imgs = [cv2.imread(f) for f in picks]
if imgs and all(i is not None for i in imgs):
    target_w = 480
    h = imgs[0].shape[0] * target_w // imgs[0].shape[1]
    imgs = [cv2.resize(i, (target_w, h)) for i in imgs]
    cv2.imwrite("$DEMO_OUT/demo_frames_summary.png", np.hstack(imgs))
    print("Wrote demo_frames_summary.png")
else:
    print("Not enough frames to build a strip")
PYEOF
    if [[ -f "$DEMO_OUT/demo_frames_summary.png" ]]; then
        ok "Summary strip  →  $DEMO_OUT/demo_frames_summary.png"
    fi
fi

# -----------------------------------------------------------------------------
# 5. Summary index — a top-level README of what got produced
# -----------------------------------------------------------------------------
banner "5. Build run summary"

INDEX="$OUT/INDEX.md"
{
    echo "# catch_robot run — $STAMP"
    echo ""
    echo "Generated by \`bash run_all.sh\` on $(date)."
    echo ""
    echo "Configuration:"
    echo "- quick mode      : $QUICK"
    echo "- skip training   : $SKIP_TRAIN"
    echo "- skip vision exp : $SKIP_VISION"
    echo "- experiments run : \`$EXP_LIST\`"
    echo "- full grids      : $([[ -n "$FULL_FLAG" ]] && echo yes || echo no)"
    echo "- total time      : $(elapsed)"
    echo ""
    echo "## Files"
    echo ""
    echo "### Plots ($OUT/plots/)"
    find "$OUT/plots" -name "*.png" 2>/dev/null | sort | sed "s|^$OUT/||" | sed 's/^/- /'
    echo ""
    echo "### Demo ($OUT/demo/)"
    find "$OUT/demo" -maxdepth 1 -name "*.png" -o -name "*.mp4" 2>/dev/null \
        | sort | sed "s|^$OUT/||" | sed 's/^/- /'
    if [[ -d "$DEMO_OUT/frames" ]]; then
        local_count=$(ls "$DEMO_OUT/frames"/*.png 2>/dev/null | wc -l)
        echo "- demo/frames/ ($local_count extracted frames)"
    fi
    echo ""
    echo "### Experiments ($OUT/experiments/)"
    if [[ -d "$OUT/experiments" ]]; then
        for d in "$OUT/experiments"/*/; do
            [[ -d "$d" ]] || continue
            echo "- $(basename "$d")/"
            if [[ -f "$d/summary.json" ]]; then
                echo "    - summary.json"
            fi
            for csv in "$d"/progress_*.csv; do
                [[ -f "$csv" ]] && echo "    - $(basename "$csv")"
            done
        done
    fi
    echo ""
    echo "### LSTM ($OUT/)"
    [[ -f "$OUT/lstm_predictor.pt" ]] && echo "- lstm_predictor.pt"
    [[ -f "$OUT/training_curve.png" ]] && echo "- training_curve.png"
    echo ""
    echo "## Headline numbers"
    echo ""
    # Pull catch rates out of each experiment's summary.json
    for d in "$OUT/experiments"/*/; do
        if [[ -f "$d/summary.json" ]]; then
            $PYTHON - <<PYEOF
import json, pathlib
p = pathlib.Path("$d/summary.json")
data = json.loads(p.read_text())
res = data.get("results", {})
for key, val in res.items():
    if not isinstance(val, dict):
        continue
    if key == "A":
        cr = val.get("catch_rate")
        if cr is not None:
            print(f"- **Experiment A** (velocity sweep): "
                  f"catch rate = {cr*100:.0f}%   "
                  f"n_runs = {val.get('n_runs')}   "
                  f"mean min-dist = {val.get('mean_min_dist_cm'):.1f} cm   "
                  f"mean first-intercept = {val.get('mean_first_intercept_ms'):.0f} ms")
    elif key == "B":
        per = val.get("per_level", {})
        print(f"- **Experiment B** (noise robustness):")
        for sigma_str, stats in per.items():
            sigma = float(sigma_str)
            print(f"    - σ = {sigma*1000:5.1f} mm  →  "
                  f"catch_rate {stats['catch_rate']*100:.0f}%, "
                  f"median dist {stats['median_dist_cm']:.1f} cm")
    elif key == "C":
        print(f"- **Experiment C** (intercept timeline): "
              f"caught = {val.get('caught')}, "
              f"min dist = {val.get('min_dist_cm'):.1f} cm, "
              f"#commits = {val.get('n_intercept_commits')}")
    elif key == "D":
        if val.get("skipped"):
            print(f"- **Experiment D** (predictor ablation): SKIPPED "
                  f"({val.get('reason')})")
        else:
            ra = val.get("catch_rate_analytical") * 100
            rl = val.get("catch_rate_learned") * 100
            print(f"- **Experiment D** (predictor ablation): "
                  f"analytical = {ra:.0f}%, learned = {rl:.0f}%")
    elif key == "E":
        cr = val.get("catch_rate", 0)
        print(f"- **Experiment E** (real-vision spot check): "
              f"catch rate = {cr*100:.0f}%, n_runs = {val.get('n_runs')}")
    elif key == "F":
        counts = val.get("counts", {})
        total = sum(counts.values())
        line = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"- **Experiment F** (failure modes, n={total}): {line}")
PYEOF
        fi
    done
} > "$INDEX"

ok "Run summary  →  $INDEX"

# -----------------------------------------------------------------------------
banner "DONE — total time: $(elapsed)"
echo "  Everything is under:   $OUT/"
echo "  Top-level summary  :   $OUT/INDEX.md"
echo "  Plots              :   $OUT/plots/"
echo "  Experiments        :   $OUT/experiments/"
echo "  Demo video & frames:   $OUT/demo/"
echo "  Full log           :   $OUT/run_all.log"
echo ""
echo "Tip: \`tar -czf catch_robot_run_${STAMP}.tar.gz -C results ${STAMP}\`"
echo "     bundles this entire run into a single tarball."
