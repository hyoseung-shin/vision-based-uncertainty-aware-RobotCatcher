#!/usr/bin/env bash
# =============================================================================
# run_viewer.sh — catch_robot 을 MuJoCo Viewer 로 열고 테스트하는 단일 진입점
# =============================================================================
# ★ 재구성 노트 (v2)
#   - run_all.sh 와 동일한 컨벤션을 따른다:
#       PYTHON="${PYTHON:-python}" 으로 인터프리터 지정 가능,
#       banner/ok/warn 출력 스타일, set -euo pipefail.
#   - live 모드는 프로젝트의 scripts/view_live.py 파이프라인을
#     scripts/view_scene.py 래퍼를 통해 실행한다.
#   - run_all.sh 가 export 하는 MUJOCO_GL=egl 이 셸에 남아 있어도
#     뷰어 창이 뜨도록 live/static 모드에서는 GLFW 로 강제한다.
#   - 압축 해제 위치가 어디든 동작하도록 프로젝트 루트를 자동 탐색한다
#     (스크립트 위치 → ./catch_robot/ 하위 순).
#
# USAGE
# -----
#   ./run_viewer.sh                          # live: 캐치 파이프라인 실시간 뷰어
#   ./run_viewer.sh live --use-gt --auto-reset
#   ./run_viewer.sh live --experiment --n-throws 20 --seed 1
#   ./run_viewer.sh static                   # 씬 XML 만 열기
#                                            # (python -m mujoco.viewer 와 동일)
#   ./run_viewer.sh check                    # 헤드리스 1회 스모크 테스트
#   ./run_viewer.sh check --vy -2.8 --vz 1.2
#
# live 모드 추가 인자는 view_live.py 의 옵션을 그대로 따른다:
#   --use-gt --vy --vz --auto-reset --experiment --n-throws --seed
# =============================================================================

set -euo pipefail

# ---- 출력 헬퍼 (run_all.sh 스타일) ------------------------------------------
BOLD="$(printf '\033[1m')"; GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"; RED="$(printf '\033[31m')"
RESET="$(printf '\033[0m')"
ok()   { printf "${GREEN}  ✓${RESET} %s\n" "$1"; }
warn() { printf "${YELLOW}  ⚠${RESET} %s\n" "$1"; }
fail() { printf "${RED}  ✗${RESET} %s\n" "$1"; }

# ---- 프로젝트 루트 탐색 ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=""
for cand in "$SCRIPT_DIR" "$SCRIPT_DIR/catch_robot"; do
  if [[ -f "$cand/catch_robot.py" && -f "$cand/scripts/view_live.py" ]]; then
    PROJECT_ROOT="$cand"; break
  fi
done
if [[ -z "$PROJECT_ROOT" ]]; then
  fail "프로젝트 루트를 찾지 못했습니다 (catch_robot.py / scripts/view_live.py 기준)."
  echo "  이 스크립트를 catch_robot.py 가 있는 폴더(또는 그 한 단계 위)에 두세요." >&2
  exit 1
fi
cd "$PROJECT_ROOT"

SCENE="assets/franka_emika_panda/catch_scene.xml"
VENV_DIR=".venv"

# ---- 모드 파싱 ----------------------------------------------------------------
MODE="live"
if [[ $# -gt 0 ]]; then
  case "$1" in
    static|live|check) MODE="$1"; shift ;;
  esac
fi

# ---- Python 선택 --------------------------------------------------------------
# 우선순위: $PYTHON 환경변수 → python → python3
#  - Windows 의 Microsoft Store 스텁(WindowsApps)은 제외
#  - 후보 중 mujoco 가 import 되는 인터프리터가 있으면 그것을 최우선 선택
is_usable_python() {
  local exe="$1" p
  command -v "$exe" >/dev/null 2>&1 || return 1
  p="$(command -v "$exe")"
  case "$p" in */WindowsApps/*) return 1 ;; esac    # MS Store 스텁 제외
  "$exe" --version >/dev/null 2>&1
}

USER_PY="${PYTHON:-}"      # run_all.sh 와 동일하게 PYTHON 환경변수로 지정 가능
PYTHON=""
FALLBACK=""
for cand in "$USER_PY" python python3; do
  [[ -n "$cand" ]] || continue
  is_usable_python "$cand" || continue
  if "$cand" -c "import mujoco" >/dev/null 2>&1; then
    PYTHON="$cand"; break                            # mujoco 보유 → 즉시 채택
  fi
  [[ -z "$FALLBACK" ]] && FALLBACK="$cand"
done
[[ -n "$PYTHON" ]] || PYTHON="$FALLBACK"
if [[ -z "$PYTHON" ]]; then
  fail "사용 가능한 python 을 찾지 못했습니다 (MS Store 스텁만 감지됨)."
  echo "  conda 환경을 활성화하거나, PYTHON=/path/to/python 으로 지정하세요." >&2
  exit 1
fi
echo "[run_viewer] python = $(command -v "$PYTHON")  ($("$PYTHON" --version 2>&1))"

# mujoco 가 없으면 .venv 를 만들어 requirements.txt 설치
if "$PYTHON" -c "import mujoco" >/dev/null 2>&1; then
  PY="$PYTHON"
  ok "mujoco $($PY -c 'import mujoco; print(mujoco.__version__)') (현재 환경)"
else
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[run_viewer] 가상환경 생성: $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  # 플랫폼별 venv 인터프리터 경로 (Windows: Scripts/, Unix: bin/)
  if [[ -x "$VENV_DIR/Scripts/python.exe" || -f "$VENV_DIR/Scripts/python.exe" ]]; then
    PY="$VENV_DIR/Scripts/python"
  else
    PY="$VENV_DIR/bin/python"
  fi
  if [[ ! -f "$PY" && ! -f "$PY.exe" ]]; then
    fail "venv 인터프리터를 찾지 못했습니다: $VENV_DIR (삭제 후 재시도: rm -rf $VENV_DIR)"
    exit 1
  fi
  if ! "$PY" -c "import mujoco" >/dev/null 2>&1; then
    echo "[run_viewer] 의존성 설치 (requirements.txt)..."
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r requirements.txt
  fi
  ok "mujoco $($PY -c 'import mujoco; print(mujoco.__version__)') (.venv)"
fi

[[ -f "$SCENE" ]] || { fail "씬 파일이 없습니다: $SCENE"; exit 1; }

# ---- GL 백엔드 ----------------------------------------------------------------
# run_all.sh 는 헤드리스 렌더링을 위해 MUJOCO_GL=egl 을 export 한다.
# 윈도우 뷰어(live/static)는 GLFW 가 필요하므로 여기서 덮어쓴다.
if [[ "$MODE" != "check" ]]; then
  export MUJOCO_GL="${VIEWER_GL:-glfw}"
fi

# ---- macOS: 윈도우 뷰어는 mjpython 필요 ----------------------------------------
RUNNER="$PY"
if [[ "$(uname -s)" == "Darwin" && "$MODE" != "check" ]]; then
  PY_BIN_DIR="$(dirname "$(command -v "$PY" || echo "$PY")")"
  if [[ -x "$PY_BIN_DIR/mjpython" ]]; then
    RUNNER="$PY_BIN_DIR/mjpython"
  elif command -v mjpython >/dev/null 2>&1; then
    RUNNER="$(command -v mjpython)"
  else
    warn "macOS 인데 mjpython 을 찾지 못했습니다. live 뷰어가 실행되지 않을 수 있습니다."
  fi
fi

# ---- 실행 ----------------------------------------------------------------------
printf "${BOLD}[run_viewer]${RESET} mode=%s  root=%s\n" "$MODE" "$PROJECT_ROOT"
case "$MODE" in
  static) exec "$RUNNER" scripts/view_scene.py --mode static "$@" ;;
  live)   exec "$RUNNER" scripts/view_scene.py --mode live   "$@" ;;
  check)  exec "$PY"     scripts/view_scene.py --mode check  "$@" ;;
esac