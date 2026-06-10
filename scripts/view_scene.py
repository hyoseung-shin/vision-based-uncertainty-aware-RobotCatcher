from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCENE_XML = ROOT / "assets" / "franka_emika_panda" / "catch_scene.xml"
PHYSICS_HZ = 500
CATCH_DIST_THRESH = 0.10


def resolve_scene_path(scene: str | Path) -> str:
    scene = Path(scene).resolve()
    s = str(scene)
    try:
        s.encode("ascii")
        return s 
    except UnicodeEncodeError:
        pass

    import hashlib
    import shutil
    import tempfile

    src_dir = scene.parent            # .../assets/franka_emika_panda
    tag = hashlib.md5(str(src_dir).encode("utf-8")).hexdigest()[:8]
    dst_dir = Path(tempfile.gettempdir()) / f"catch_robot_scene_{tag}"
    dst_scene = dst_dir / scene.name

    try:
        str(dst_dir).encode("ascii")
    except UnicodeEncodeError:
        print("[view_scene] 경고: 임시 디렉터리 경로도 비ASCII 입니다. "
              "프로젝트를 ASCII 경로(예: C:\\dev\\catch_robot)로 옮겨 주세요.", file=sys.stderr)
        return s

    need_copy = (not dst_scene.exists() or dst_scene.stat().st_mtime < scene.stat().st_mtime)
    if need_copy:
        print(f"[view_scene] 비ASCII 경로 감지 → 모델 폴더를 임시 ASCII 경로로 "
              f"미러링: {dst_dir}")
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    return str(dst_scene)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="catch_robot MuJoCo viewer launcher (view_live 래퍼)",
        epilog="--mode live 일 때 나머지 인자는 scripts/view_live.py 로 전달됩니다.")
    p.add_argument("--mode", choices=["static", "live", "check"], default="live")
    p.add_argument("--scene", default=str(SCENE_XML), help="static 모드에서 열 씬 XML 경로")
    p.add_argument("--duration", type=float, default=1.5, help="check 모드: 시뮬레이션 시간 [s]")
    p.add_argument("--vy", type=float, default=-2.5, help="공 초기 y 속도 [m/s] (view_live 기본값과 동일)")
    p.add_argument("--vz", type=float, default=1.5, help="공 초기 z 속도 [m/s] (view_live 기본값과 동일)")
    return p.parse_known_args()


def run_static(scene_xml: str) -> int:
    import mujoco
    import mujoco.viewer
    if not Path(scene_xml).exists():
        print(f"[view_scene] 씬 파일을 찾을 수 없습니다: {scene_xml}", file=sys.stderr)
        return 2
    scene_xml = resolve_scene_path(scene_xml)
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_ready")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
    print(f"[view_scene] 인터랙티브 뷰어 실행: {scene_xml}")
    print("[view_scene] 창을 닫으면 종료됩니다.")
    mujoco.viewer.launch(model, data)
    return 0


def run_live(passthrough: list[str], args: argparse.Namespace) -> int:
    if sys.platform == "darwin" and "mjpython" not in sys.executable:
        print("[view_scene] 경고: macOS 에서 live 모드는 mjpython 으로 실행해야 "
              "합니다.\n  예) mjpython scripts/view_scene.py --mode live\n"
              "  (run_viewer.sh 사용 시 자동 처리)", file=sys.stderr)

    from scripts import view_live
    view_live.SCENE = resolve_scene_path(view_live.SCENE)
    argv = ["--vy", str(args.vy), "--vz", str(args.vz)] + passthrough
    old_argv = sys.argv
    sys.argv = [str(ROOT / "scripts" / "view_live.py")] + argv
    try:
        view_live.main()
    finally:
        sys.argv = old_argv
    return 0


def run_check(args: argparse.Namespace, passthrough: list[str]) -> int:
    import numpy as np
    from scripts import view_live
    from scripts.view_live import LiveRunner
    view_live.SCENE = resolve_scene_path(view_live.SCENE) 

    use_gt = "--use-gt" in passthrough or True
    runner = LiveRunner(use_gt=use_gt, vy=args.vy, vz=args.vz)

    n_steps = int(args.duration * PHYSICS_HZ)
    min_dist = float("inf")
    t0 = time.perf_counter()
    for i in range(n_steps):
        runner.step_once(i)
        d = float(np.linalg.norm(runner.data.xpos[runner.ball_id] - runner.data.xpos[runner.ee_id]))
        min_dist = min(min_dist, d)
    dt = time.perf_counter() - t0

    caught = min_dist < CATCH_DIST_THRESH and runner.committed_catch_t is not None
    print(f"[check] {n_steps} physics steps in {dt:.2f}s ({n_steps/dt:.0f} Hz)")
    print(f"[check] throw vy={args.vy:+.2f} vz={args.vz:+.2f}  "
          f"min dist = {min_dist*100:.1f} cm  "
          f"committed = {runner.committed_catch_t is not None}")
    print(f"[check] result: {'CATCH' if caught else 'MISS'}")
    return 0 if caught else 1


def main() -> int:
    args, passthrough = parse_args()

    if args.mode == "check":
        return run_check(args, passthrough)

    os.environ["MUJOCO_GL"] = os.environ.get("VIEWER_GL", "glfw")

    if args.mode == "static":
        return run_static(args.scene)
    return run_live(passthrough, args)


if __name__ == "__main__":
    raise SystemExit(main())