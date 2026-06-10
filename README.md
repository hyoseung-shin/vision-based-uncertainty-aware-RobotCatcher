# catch_robot — 비전 기반 동적 객체 캐치 로봇

> MuJoCo 시뮬레이션에서 **스테레오 카메라만으로** 던져진 공을 인지·추적·예측하고,
> **Franka Emika Panda 7-DOF 매니퓰레이터**로 실시간에 잡아내는 통합 시스템.
>
> 로봇메카트로닉스(SWE4037) 최종 프로젝트 · 대상 직무: NAVER LABS Robotics Engineer (Perception / Manipulation)

```
스테레오 RGB (60 Hz) → HSV 컬러 검출 → DLT 삼각측량(공분산 전파) →
Mahalanobis 게이팅 트랙 → 가속도 EKF → RK4+항력 궤적 예측 →
불확실성 가중 요격점 선택 → DLS-IK → 부드러운 관절 제어 (500 Hz)
```

**검증된 핵심 결과** — GT 모드 catch rate **75 %** (60-throw sweep), 스테레오 3D 정확도 **9.4 mm**, 비전 노이즈 강건성 임계점 **σ ≈ 30 mm**.

---

## 목차

1. [데모 미리보기](#1-데모-미리보기)
2. [환경 설정 (OS별)](#2-환경-설정-os별)
3. [빠른 시작 — 30초 검증](#3-빠른-시작--30초-검증)
4. [실시간 뷰어로 실험 관전하기 (`run_viewer.sh`)](#4-실시간-뷰어로-실험-관전하기-run_viewersh)
5. [실험 스위트 A–F (`run_experiments.py`)](#5-실험-스위트-af-run_experimentspy)
6. [전체 파이프라인 한 번에 (`run_all.sh`)](#6-전체-파이프라인-한-번에-run_allsh)
7. [단일 throw 실행 (`catch_robot.py`)](#7-단일-throw-실행-catch_robotpy)
8. [산출물 정리](#8-산출물-정리)
9. [프로젝트 구조](#9-프로젝트-구조)
10. [4개 핵심 주제 매핑](#11-4개-핵심-주제-매핑)
11. [라이선스](#12-라이선스)

---

## 1. 데모 미리보기

| 모드 | 명령 | 보게 되는 것 |
|---|---|---|
| 실시간 뷰어 (관전) | `./run_viewer.sh live --use-gt --experiment` | MuJoCo 창에서 로봇이 무작위 공을 연속으로 잡는 모습 + 콘솔 누적 통계 |
| 정적 씬 확인 | `./run_viewer.sh static` | `catch_scene.xml` 의 로봇·카메라 리그·공 배치 |
| 영상 파일 (헤드리스) | `python scripts/demo.py --use-gt` | `demo_overview.mp4` (50 fps 캐치 영상) |

---

## 2. 환경 설정 (OS별)

### 공통 요구사항

- Python **3.10+** (3.12 권장)
- `pip install -r requirements.txt` — mujoco(≥3.2), numpy, opencv-python-headless, matplotlib, PyOpenGL
- 선택: `pip install torch` (LSTM 예측기 학습/비교 실험 D 에만 필요)
- GPU 불필요 (헤드리스 렌더링은 EGL 소프트웨어 폴백으로 동작)

### 2.1. Linux

```bash
git clone <repo-url> && cd catch_robot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_viewer.sh check          # 헤드리스 스모크 테스트
./run_viewer.sh                # 데스크톱이면 뷰어 창 실행
```

- 헤드리스 서버에서는 `export MUJOCO_GL=egl` 후 `scripts/demo.py` 로 영상 출력.
- 데스크톱(X11/Wayland)에서는 추가 설정 없이 뷰어가 뜬다.

### 2.2. macOS

```bash
git clone <repo-url> && cd catch_robot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_viewer.sh check
./run_viewer.sh                # mjpython 을 자동 탐지해 사용
```

- macOS 의 MuJoCo 창 뷰어(`launch_passive`)는 **`mjpython`** 으로 실행해야 한다.
  `mjpython` 은 mujoco 패키지 설치 시 함께 들어오며, `run_viewer.sh` 가 자동으로 찾아 사용한다.
- 수동 실행 시: `mjpython scripts/view_live.py --use-gt`

### 2.3. Windows (Git Bash + Anaconda 기준)

```bash
# Anaconda Prompt 또는 Git Bash 에서:
conda create -n mujoco_project python=3.10
conda activate mujoco_project
pip install -r requirements.txt

# Git Bash 에서 프로젝트 폴더로 이동 후:
bash run_viewer.sh check
bash run_viewer.sh             # 뷰어 창 실행 (mjpython 불필요)
```

Windows 특이사항 — `run_viewer.sh` 가 모두 자동 처리하지만 알아두면 좋다:

| 이슈 | 증상 | 처리 방식 |
|---|---|---|
| **Microsoft Store 스텁** | `python3` 가 `WindowsApps` 의 가짜 인터프리터로 잡혀 즉시 멈춤 | 스크립트가 `WindowsApps` 경로를 후보에서 제외하고, **mujoco 가 import 되는 인터프리터를 우선 선택** |
| **venv 레이아웃 차이** | Windows 는 `.venv/Scripts/python.exe` (Unix 는 `.venv/bin/python`) | 두 경로 모두 자동 판별 |
| **비ASCII 경로** | 한글 폴더("바탕 화면" 등)에서 MuJoCo 가 `ParseXML: Error opening file` | `view_scene.py` 가 모델 폴더를 `%TEMP%` 의 ASCII 경로로 1회 미러링 후 로드 (이후 캐시 재사용) |

> **권장**: 가능하면 프로젝트를 `C:\dev\catch_robot` 처럼 **공백·한글 없는 경로**에 두자.
> MuJoCo 외에도 많은 C/C++ 기반 도구가 비ASCII 경로에서 문제를 일으키고,
> OneDrive 동기화 폴더는 실험 산출물(`data/`, `results/`) 업로드로 I/O 가 느려진다.
> WSL2 에서 Linux 절차를 따르는 것도 좋은 대안이다.

---

## 3. 빠른 시작 — 30초 검증

```bash
python scripts/test_modules.py
```

기대 출력:

```
Running 5 module tests:
  EKF gravity estimate: a_z = -10.34   PASS
  Predictor end-of-horizon error: 20.4 mm   PASS
  Tracker rejects outlier: misses=5   PASS
  IK reaches 5 targets sub-cm: 5/5   PASS
  Intercept finds a reachable catch point: PASS
5/5 tests passed.
```

5/5 PASS 면 전체 시스템이 정상 동작 가능한 상태다. 이어서:

```bash
./run_viewer.sh check     # 캐치 파이프라인 1회 헤드리스 실행 → CATCH 기대
```

---

## 4. 실시간 뷰어로 실험 관전하기 (`run_viewer.sh`)

`run_viewer.sh` 는 뷰어 관련 작업의 **단일 진입점**이다. 내부적으로
`scripts/view_scene.py` → `scripts/view_live.py` 로 위임하며, Python/venv 탐색,
OS별 차이(mjpython, venv 경로, 비ASCII 경로), GL 백엔드 설정을 모두 처리한다.

```bash
./run_viewer.sh [static|live|check] [추가 옵션...]
```

| 모드 | 설명 |
|---|---|
| `live` (기본) | 뷰어 창에서 캐치 파이프라인을 실시간 실행 |
| `static` | 씬 XML 만 인터랙티브 뷰어로 열기 (`python -m mujoco.viewer --mjcf=...` 와 동일) |
| `check` | 뷰어/GL 없이 1회 실행 후 CATCH/MISS 출력 (헤드리스, CI 용) |

### 자주 쓰는 조합

```bash
# 한 번 던지기 (기본). BACKSPACE 로 수동 재던지기
./run_viewer.sh live --use-gt

# ★ 진행 중인 실험 관전: 무작위 throw 연속 + 콘솔 누적 통계
./run_viewer.sh live --use-gt --experiment

# 같은 조건으로 1초 간격 자동 재던지기 무한 반복
./run_viewer.sh live --use-gt --auto-reset

# 20회 던지고 자동 종료 (최종 catch rate / 평균 거리 출력, 시드 고정)
./run_viewer.sh live --use-gt --experiment --n-throws 20 --seed 1

# 실제 스테레오 비전 파이프라인으로 관전 (느리지만 "진짜" 실험)
./run_viewer.sh live --auto-reset

# 던지기 조건 변경 / 헤드리스 검증
./run_viewer.sh live --use-gt --vy -2.8 --vz 1.2
./run_viewer.sh check --vy -2.8 --vz 1.2
```

`--experiment` 모드 콘솔 출력 예:

```
[throw   1]  caught=True   final_dist=6.7cm  vy=-2.34 vz=+1.42  →  running catch_rate=100%  mean_dist=6.7cm
[throw   2]  caught=True   final_dist=5.9cm  vy=-2.78 vz=+1.85  →  running catch_rate=100%  mean_dist=6.3cm
[throw   3]  caught=False  final_dist=14.2cm vy=-2.13 vz=+0.92  →  running catch_rate=67%   mean_dist=8.9cm
```

**뷰어 조작**: 마우스 드래그 = 회전, 휠 = 줌, 우클릭 드래그 = 이동,
`SPACE` = 일시정지/재개, `BACKSPACE` = 키프레임 리셋 + 재던지기.

**`--use-gt` 의 의미**: 스테레오 렌더링을 우회하고 GT 공 위치 + 가우시안 노이즈(σ=10 mm)를
EKF 에 공급한다. 제어·예측 파이프라인은 동일하게 동작하면서 뷰어가 항상 실시간으로 부드럽다.
빼면 매 비전 틱마다 카메라 2대를 오프스크린 렌더링하는 전체 인지 파이프라인이 돈다
(GPU 에 따라 실시간보다 느려질 수 있음).

---

## 5. 실험 스위트 A–F (`run_experiments.py`)

정량 결과는 모두 이 스위트가 생성한다. 보고서·발표 수치는 `--full` 격자 기준.

```bash
python scripts/run_experiments.py              # 기본 세트 A+B+C (~15초)
python scripts/run_experiments.py --full       # 본격 격자 (~3분)
python scripts/run_experiments.py --exp F --full   # 개별 실험 선택
```

| 실험 | 내용 | 산출물 | `--full` 규모 |
|---|---|---|---|
| **A** | 초기 속도 (vx,vy,vz) 격자 catch rate | `catch_rate_by_velocity.png` | 60 throws |
| **B** | 비전 노이즈 σ 강건성 (0–80 mm) | `error_vs_noise.png` | 72 throws |
| **C** | 대표 throw 의 시간별 거동 | `intercept_timeline.png` | 1 throw |
| **D** | Analytical vs LSTM 예측기 ablation | `predictor_ablation.png` | (ckpt 필요) |
| **E** | 실제 스테레오 렌더 파이프라인 검증 | `progress_E.csv` | GPU 30초 / CPU 3분 |
| **F** | 실패 5종 분류 (near_miss, ee_diverged, …) | `failure_modes.png` | 60 throws |

진행 상황은 throw 단위로 즉시 CSV 에 flush 되므로 다른 터미널에서 실시간 모니터링 가능:

```bash
tail -f data/experiments/$(ls -t data/experiments/ | head -1)/progress_A.csv
```

LSTM 예측기(D 실험)는 다음 3단계로 준비한다:

```bash
python scripts/collect_data.py --n-throws 2000   # 궤적 수집 (~2분)
python scripts/train_predictor.py --epochs 30    # 학습 → data/lstm_predictor.pt
python scripts/run_experiments.py --exp D --full
```

---

## 6. 전체 파이프라인 한 번에 (`run_all.sh`)

단위 테스트 → (선택) LSTM 학습 → 실험 A–F → 데모 영상 → 진단 플롯을
한 명령으로 수행하고, 모든 산출물을 `results/<timestamp>/` 에 수집한다.

```bash
bash run_all.sh              # 전체 (~5분 with torch / ~3분 without)
bash run_all.sh --quick      # 스모크 테스트 (~1분): --full 격자·학습 생략
bash run_all.sh --skip-train # 학습만 생략
```

재실행해도 이전 결과를 건드리지 않는다 (타임스탬프 폴더 신규 생성).

---

## 7. 단일 throw 실행 (`catch_robot.py`)

디버깅·튜닝용 최소 진입점. JSON 결과를 stdout 으로 출력한다.

```bash
python catch_robot.py --use-gt                       # GT 모드 (~2초)
python catch_robot.py --use-gt --vy -2.5 --vz 1.5    # 던지기 조건 변경
python catch_robot.py --vy -2.5 --vz 1.5             # 전체 비전 (~80초 CPU)
python catch_robot.py --use-gt --predictor learned --learned-ckpt data/lstm_predictor.pt
```

```json
{ "caught": true, "min_dist_m": 0.067, "t_first_intercept": 0.096, "n_detections": 90 }
```

---

## 8. 산출물 정리

| 경로 | 생성 주체 | 내용 |
|---|---|---|
| `data/experiments/<timestamp>/` | `run_experiments.py` | `summary.json`, `progress_*.csv`, 실험별 PNG 플롯 |
| `data/lstm_predictor.pt` | `train_predictor.py` | 학습된 LSTM 체크포인트 |
| `data/trajectories.npz` | `collect_data.py` | 학습용 궤적 데이터 |
| `data/training_curve.png` | `train_predictor.py` | epoch 별 학습 곡선 (실시간 갱신) |
| `results/<timestamp>/` | `run_all.sh` | 위 전부 + `run_all.log` + 데모 영상의 통합 사본 |
| `demo_overview.mp4`, `demo_stats.png` | `scripts/demo.py` | 발표용 캐치 영상·통계 |

대표 수치 (`--full`, GT 모드): catch rate 75 %, 평균 최소거리 ~8 cm,
첫 intercept commit ~100 ms, 스테레오 3D 오차 평균 9.4 mm (`smoke_test_vision.py`).

---

## 9. 프로젝트 구조

```
catch_robot/
├── run_viewer.sh                  ★ 뷰어 단일 진입점 (OS별 처리 포함)
├── run_all.sh                       전체 파이프라인 (테스트→학습→실험→데모)
├── catch_robot.py                   단일 throw 런타임 (CatchRunner, CLI)
├── requirements.txt
├── README.md                        이 파일
│
├── assets/franka_emika_panda/       MuJoCo Menagerie Franka 모델 (수정 없음)
│   ├── catch_scene.xml              ★ 메인 씬: 스테레오 리그·공·키프레임 정의
│   ├── panda.xml · scene.xml · mjx_*.xml
│   └── assets/                      메시 73개 (.obj / .stl)
│
├── vision/                          인지 파이프라인
│   ├── camera.py                    핀홀 모델, StereoRig 오프스크린 렌더
│   ├── detector.py                  HSV 컬러 공 검출
│   ├── stereo.py                    DLT 삼각측량 + 공분산 전파
│   ├── tracker.py                   Mahalanobis 게이팅 트래커
│   ├── ekf.py                       9-state 가속도 EKF
│   ├── predictor.py                 Analytical(RK4+항력) / Learned 통합
│   ├── predictor_model.py           LSTM 정의 (lazy torch import)
│   └── intercept.py                 불확실성 가중 catch point 선택
│
├── planning/                        계획·제어
│   ├── ik.py                        Damped Least-Squares IK
│   └── controller.py                JointTracker (관절 보간 + 그리퍼)
│
├── scripts/
│   ├── view_scene.py              ★ 뷰어 런처 (static/live/check, 비ASCII 경로 우회)
│   ├── view_live.py                 실시간 뷰어 파이프라인 (+experiment 모드)
│   ├── run_experiments.py           ★ 메인 실험 스위트 A–F
│   ├── test_modules.py              단위 테스트 5종 (가장 먼저 실행)
│   ├── smoke_test_vision.py         비전 정확도 단독 검증
│   ├── collect_data.py / train_predictor.py   LSTM 데이터·학습
│   ├── demo.py                      헤드리스 영상 출력
│   └── throw_ball.py                공 초기 상태 헬퍼
│
├── data/                            학습 데이터·체크포인트·실험 결과
└── results/                         run_all.sh 타임스탬프별 통합 산출물
```

실행 흐름: `run_viewer.sh` → `view_scene.py` → `view_live.py` → `vision/` + `planning/`,
씬은 `assets/franka_emika_panda/catch_scene.xml` 하나로 모인다.

---

## 10. 4개 핵심 주제 매핑

| 주제 | 구현 위치 | 핵심 문장 |
|---|---|---|
| **Kinematics** | `planning/ik.py` (DLS-IK), `vision/intercept.py` 의 FK 기반 도달성 | 검출 픽셀 → 카메라 → world → EE 목표로 변환되고, Jacobian pseudo-inverse 가 관절 명령을 생성 |
| **Dynamics** | `vision/ekf.py` 가속도 추정, `vision/predictor.py` 중력+항력 RK4 | EKF 가 데이터에서 −9.81 m/s² 를 학습하고, predictor 가 0.5 s 미래를 20 mm 이내로 예측 |
| **Robot Vision** | `vision/{camera,detector,stereo,tracker}.py` | 스테레오 픽셀 → 3D + 공분산. 깊이 방향 σ가 측면의 최대 20배인 비등방성을 측정·활용 |
| **Path Planning** | `vision/intercept.py` + 60 Hz 재계획 | 예측 불확실성 밴드 ∩ 도달 영역에서 비용 최소화로 catch point 선택, 5 cm 이상 변화 시에만 re-commit |

---

## 11. 라이선스

- `assets/franka_emika_panda/` — Apache 2.0 ([MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie))
- 본 프로젝트 코드 — MIT License