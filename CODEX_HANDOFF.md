# 다른 PC의 Codex 인계: 4차 RL 과제

이 문서는 이전 대화 없이도 환경 준비 → 신규 모델 두 개 학습 → 그래프/주행 확인까지 진행할 수 있는 인계서다.
작업 저장소는 **https://github.com/sanghyeonjung/AI_mobility** 이다.
사용자가 다른 PC에서 이 문서를 읽고 진행하라고 요청하면 아래 작업을 수행한다.
현재 문서를 작성한 PC에서는 과제 모델 학습을 실행하지 않았다.

## 1. 사용자 요청과 완료 기준

- 수업 실습 `4_Model_Training_RL.md`의 PPO를 사용해 **새 모델 A와 B를 각각 학습**한다.
- GitHub에 원래 있던 모델, BC 모델, 기존 학습 가중치를 두 모델 중 하나로 포함하지 않는다.
- **총 학습량 `TOTAL_TIMESTEPS`는 두 모델 모두 200,000으로 유지한다.**
- 모델 사이에서는 **학습률 `lr`만 바꾼다.** 도로·보상·관측·행동·나머지 학습 설정을 동일하게 유지한다.
- TensorBoard에서 새 A/B run의 학습 그래프를 함께 확인하고, 학습된 정책을 SUMO 도로에서 주행시킨다.
- 사진의 공지 기준 마감은 **2026년 10월 12일 23:59, 한국 시간**이다.
- 실제 최초 학습 시작 시각이 수업 이후여야 한다. 클래스 종료 시각은 이 문서에 제공되지 않았다.
  적격 시각 확인이 필요하면 사용자에게 실제 수업 종료 시각을 확인한다. 시간이나 run 이름을 조작하지 않는다.
- **제출 사진과 학과·학번·이름 메모는 사용자가 직접 준비한다.** 개인정보를 요청하거나 임의 작성하지 않는다.

완료 시에는 각 모델의 실제 run 경로, 학습 스텝, `model.pt`/`checkpoint.pt`/TensorBoard 로그 존재 여부,
학습률 차이, 실제 측정된 평가 결과와 캡처 화면 열기 방법을 사용자에게 보고한다.
학습을 끝내지 않았다면 완료했다고 말하지 않는다. 실측 그래프나 성능 수치를 만들어 넣지 않는다.

## 2. 변경할 것과 고정할 것

| 설정 | 모델 A | 모델 B |
|---|---:|---:|
| `lr` | `5e-5` = `0.00005` | `3e-4` = `0.0003` |
| `TOTAL_TIMESTEPS` | **200,000** | **200,000** |
| `n_steps` | 2,048 | 2,048 |
| `n_epochs` | 10 | 10 |
| `minibatch_size` | 64 | 64 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 동일 |
| `hidden_sizes` / `activation` | (256, 256) / ReLU | 동일 |
| `policy_type` / 초기 seed | hybrid / 0 | 동일 |
| 평가 간격 / 평가 에피소드 | 5 업데이트 / 5 에피소드 | 동일 |
| 도로·교통·보상·관측·행동 | 저장소의 현재 실습 설정 | 동일 |

코드는 `train.py`의 설정을 읽고 `train_assignment4.py`의 `PROFILES`로 학습률만 덮어쓴다.
원본 `train.py`의 학습률은 `1e-4`(`0.0001`)이다. A는 더 낮은 `5e-5`, B는 더 높은 `3e-4`이므로 **둘 다 원본과 다른 학습률**이다.
A도 기존 GitHub 모델을 불러오는 것이 아니라 새로운 무작위 초기화로 시작한다.
B도 독립적으로 초기화한다. 동일한 seed는 비교 조건을 맞추기 위한 것이며 학습된 가중치 재사용이 아니다.
학습률 이외의 파라미터는 사용자의 추가 요청 없이 실험 중 변경하지 않는다.

용어를 구분한다:

- **`TOTAL_TIMESTEPS`**: 목표 학습 환경 스텝 수. 두 모델 모두 200,000으로 고정한다.
- **timestamp**: run 이름과 로그에 기록되는 실제 시작 날짜·시간. 실제 컴퓨터 시각으로 기록하며 조작하지 않는다.
- **`--session-updates`**: 한 번 실행하고 저장·종료하기까지의 업데이트 횟수. 총 목표를 바꾸는 옵션이 아니다.

이전 안내의 `--timesteps 50000`은 사용자에게서 추가로 확인된 수업 조건에 맞지 않아 폐기했다.
현재 실행 코드에서는 `--timesteps` 옵션을 제거했다. 시간 부족을 이유로 학습량을 줄이지 않는다.

rollout을 2,048 스텝 단위로 모으므로 200,000 목표의 실제 종료는 **200,704 스텝, 98 업데이트**다.
이는 실습의 기존 루프 동작이며 `TOTAL_TIMESTEPS`를 200,704로 수정하지 않는다.

## 3. 첫 작업: 현재 PC와 기존 실행 확인

1. 저장소 작업 폴더와 Python/SUMO 설치 여부를 확인한다.
2. 이미 실행 중인 학습이 있으면 중복 실행하거나 강제 종료하지 않는다.
3. `results/assignment4/`에 A/B run이 있다면 `run_info.json`과 `experiment_config.json`부터 확인한다.
4. 두 run의 목표가 200,000이고 학습률이 A=`5e-5`, B=`3e-4`이며 나머지 설정이 같은지 확인한다.
5. 적합한 미완료 run이 있으면 새로 만들지 말고 재개한다. 완료된 run은 다시 학습하지 않는다.
6. 새로 clone한 저장소에는 기존 모델·BC 데이터셋·학습 결과·캐시·생성 도로 파일이 없다.
   도로 파일은 실행 시 생성되므로 다운로드한 모델이나 별도 데이터셋이 필요하지 않다.

학습이 진행 중이면 코드 업데이트를 끼워 넣지 않는다.
이전 버전에서 이미 200,000 목표로 학습을 시작했고 학습률도 위 최종 설정과 일치하면 그 버전의 코드로 해당 run을 마쳐도 된다.
새 버전으로 업데이트한 뒤 `Source files changed`가 발생하면 run의 `source/` 사본과 현재 코드를 비교한다.
필요하면 해당 run에서 사용했던 코드를 복원하되, 목표/설정이 위 조건과 일치하는지 먼저 확인한다.
해시나 체크포인트 정보를 임의 수정하여 검사를 우회하지 않는다. 기존 파일을 삭제하지 않는다.
목표를 줄였던 비적합 run만 있다면 기존 결과를 보존하고 `--new-run`으로 적합한 새 run을 만든다.

**이전 A의 학습률 `0.0001`은 이번 최종 A 설정에 맞지 않는다.**
해당 결과를 보존하고 `.\assignment4.cmd a --new-run`으로 `0.00005`의 A를 새 무작위 가중치에서 처음부터 학습한다.
이전 A를 불러온 뒤 학습률만 낮춰 이어 학습하거나 체크포인트·설정 기록을 수정하지 않는다.
B의 기존 run은 `0.0003`, 목표 200,000이며 다른 조건도 일치하면 그대로 완료해 사용할 수 있다.

## 4. Windows 환경 준비

이미 clone한 폴더가 있으면 그 폴더를 사용한다. 실행 중인 학습이 없고 새 코드를 받을 때만 `git pull --ff-only`를 사용한다.
새 PC라면 PowerShell에서:

```powershell
git clone https://github.com/sanghyeonjung/AI_mobility.git
cd AI_mobility
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-assignment4.txt
.\assignment4.cmd check
```

필요한 패키지는 `numpy`, `torch`, `gymnasium`, `traci`, `sumolib`, `tensorboard`다.
Python이 없으면 먼저 [Python 공식 사이트](https://www.python.org/downloads/)에서 설치하고 새 터미널을 연다.
SUMO는 Python 패키지와 별도 프로그램이다. 없으면 [SUMO 공식 사이트](https://eclipse.dev/sumo/)에서 Windows 인스톨러를 설치한다.
설치 시 `Set SUMO_HOME` 옵션을 선택하고 새 터미널을 연다.

`check`가 `OK: dependencies ready. No training started.`를 출력해야 실제 학습으로 넘어간다.
점검이 실패하면 에러 원인을 해결한다. 다른 Python에 패키지를 설치하고 계속 진행하지 않는다.
`.cmd`는 `ASSIGNMENT_PYTHON` 환경변수 → 프로젝트 `.venv` → Windows Python 기본 경로 → `python` 순서로 선택한다.
프로젝트 `.venv`를 쓰면 PowerShell 가상환경 활성화나 실행 정책 변경이 필요 없다.
사용할 Python을 명시하려면 현재 터미널에서:

```powershell
$env:ASSIGNMENT_PYTHON = '학습용 python.exe의 실제 절대경로'
```

SUMO가 다른 경로에 설치되어 있으면 실제 설치 폴더를 현재 터미널에 지정한다.
아래 예시는 해당 폴더가 실제 존재하는 경우에만 사용한다.

```powershell
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
.\assignment4.cmd check
```

`check`는 학습하지 않는다. 별도 점검이 필요하면 다음 테스트도 가능하다:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트는 임시 폴더를 사용하며 실제 과제 모델의 학습을 수행하지 않는다.
SUMO 연결 테스트에는 짧은 환경 초기화와 고정 행동 한 스텝이 포함된다.

## 5. 두 모델 실제 학습

짧게 나눠 실행하는 기본 방식:

```powershell
.\assignment4.cmd a
.\assignment4.cmd b
```

각 명령은 최대 10 업데이트(20,480 스텝) 후 자동 저장·종료한다.
A 명령이 끝난 뒤 B 명령을 실행한다. 동시에 두 학습을 시작하지 않는다.
같은 명령을 반복하면 최신 A/B run의 체크포인트를 이어서 사용한다.
목표는 여전히 각각 200,000이며, 각 모델을 약 10번 호출하면 98 업데이트를 마친다.
Codex가 사용자에게 계속 실행하라고 요청받았다면 중간 세션 종료를 최종 완료로 취급하지 말고 목표에 도달할 때까지 재개한다.

다른 PC를 충분히 켜둘 수 있고 사용자가 전체 학습 진행을 요청하면:

```powershell
.\assignment4.cmd a --session-updates 0
.\assignment4.cmd b --session-updates 0
```

한 번에 1 업데이트만 실행해야 하면 `--session-updates 1`을 사용한다.
세션 횟수만 바뀌며 `n_steps`, `n_epochs`, 총 목표 등은 변경하지 않는다.

진행 상태 확인:

```powershell
.\assignment4.cmd status
```

콘솔의 남은 시간 추정은 실제 측정 속도를 기반으로 하며 평가 주기에 따라 달라진다.
사전에 근거 없는 소요 시간을 단정하지 않는다.

중단은 Ctrl+C를 한 번 누르고 `Saved:`를 기다린다.
정책·optimizer·누적 스텝·난수 상태는 `checkpoint.pt`에 저장된다.
미완료 업데이트는 버리고 마지막 완료 업데이트부터 재개한다.
SUMO의 진행 중 에피소드와 최근 에피소드 통계는 재개 시 새로 시작하므로 무중단 실행과 완전히 같은 궤적은 아니다.
학습 설정과 목표는 유지된다. 첫 업데이트 전 중단한 0스텝 모델을 학습 완료 모델로 보고하지 않는다.

## 6. TensorBoard 비교와 제출 화면 준비

다른 터미널에서 실행하고 해당 서버를 유지한다:

```powershell
.\assignment4.cmd tensorboard
```

브라우저 주소는 **http://localhost:6006** 이다.
과제 전용 `results/assignment4/` 로그만 읽는다.
6006을 기존 서버가 쓰고 있다면 어떤 서버인지 확인하고, 사용자 작업을 임의 종료하지 않는다.
필요하면 사용자가 열었던 불필요한 서버를 정리하거나 별도 포트로 서버를 띄우고 실제 주소를 안내한다.

TensorBoard에서:

1. 이번 과제의 적합한 A와 B run을 모두 선택한다. 이름의 실제 날짜·시간이 보이게 한다.
2. 공지 예시와 같은 **`episode/return`, `episode/length`** 그래프를 표시한다.
3. `train/ep_rew_mean`, `eval/ep_return`도 확인한다.
4. `eval/collision_rate`, `eval/success_rate`로 주행 성능을 확인한다.
5. x축 `Step`, 두 모델의 동일한 smoothing을 사용한다.
6. Text 탭 `experiment/settings`, `experiment/started_at_kst`로 실제 설정과 시작 시각을 확인할 수 있다.

사용자는 그래프 사진에 **학과·학번·이름 메모와 현재 컴퓨터 시간**을 함께 보이게 직접 캡처한다.
시계가 안 보이면 공지처럼 별도 시계를 띄운다. JPG/JPEG/PNG/PDF로 제출할 수 있다.
Codex는 실제 로그를 화면에 표시하고 캡처할 위치를 안내한다. 개인정보 메모나 최종 제출은 요청 없이 대신 수행하지 않는다.

## 7. 학습된 차량을 도로에서 확인

```powershell
.\assignment4.cmd drive-a
.\assignment4.cmd drive-b
```

각 모델의 최신 run에서 `model.pt`를 읽고 저장된 실험 설정으로 1 에피소드를 재생한다.
SUMO GUI의 **▶ 버튼**을 눌러 주행을 시작한다. 빨간 ego 차량이 학습된 정책으로 주행한다.
GUI 조작 도구가 없으면 창을 열고 사용자에게 ▶ 버튼을 눌러 달라고 안내한다.
두 주행은 순서대로 진행한다. 사용자는 필요한 도로 주행 장면도 직접 캡처할 수 있다.

수치 비교용 5 에피소드 평가가 필요하면 실제 run 폴더를 사용한다:

```powershell
.\.venv\Scripts\python.exe test.py .\results\assignment4\실제_A_run폴더\model.pt --algorithm ppo --episodes 5 --nogui
.\.venv\Scripts\python.exe test.py .\results\assignment4\실제_B_run폴더\model.pt --algorithm ppo --episodes 5 --nogui
```

충돌률·완주율·평균 속도·차선변경·평균 보상을 실제 출력으로 비교한다.
학습률이 큰 B가 반드시 더 낫다고 가정하지 않는다. 한 쌍의 실행만으로 일반적인 우열을 단정하지 않는다.

## 8. 결과 검증과 전달

각 run 폴더에 다음을 확인한다:

```text
results/assignment4/model_a_실제시작시각/
  run_info.json             # target_steps=200000, trained_steps>=200000, status=complete
  experiment_config.json    # hparams.lr=0.00005
  model.pt                  # 주행용 가중치
  checkpoint.pt             # 재개 상태
  training_log.csv           # 마지막 steps=200704 (n_steps=2048 유지 시)
  eval_log.csv              # 실제 평가 기록
  tb/                       # 실제 TensorBoard 이벤트
  source/                   # 학습 시작 당시 코드
  sumo/                     # 생성 도로
results/assignment4/model_b_실제시작시각/
  ...                       # hparams.lr=0.0003, 나머지 조건 동일
```

A/B의 `experiment_config.json`에서 `model` 이름과 `hparams.lr`를 제외한 내용이 같아야 한다.
`run_info.json`의 두 `target_steps`는 모두 200,000이어야 한다.
분할 세션의 마지막 상태가 `paused`이면 아직 학습이 끝난 것이 아니다.
평가 로그의 마지막 스텝이 200,704보다 작을 수 있다. 평가는 5 업데이트마다라 마지막 정기 평가는 95번째 업데이트다.

학습 결과는 `.gitignore`로 Git 업로드에서 제외된다.
다른 PC로 옮길 때는 **`results/assignment4/` 전체**를 복사한다. 그래프에는 `tb/`, 재개에는 `checkpoint.pt`와 실행 당시 코드가 필요하다.
`model.pt`만 옮기면 TensorBoard 그래프나 완전한 학습 상태를 복원할 수 없다.
사용자가 요청하지 않은 학습 결과 업로드·공유·삭제·제출은 수행하지 않는다.

## 9. macOS/Linux라면

SUMO와 Python을 해당 PC에 설치한 뒤 프로젝트 폴더에서 `.cmd` 대신 다음을 사용한다:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-assignment4.txt
.venv/bin/python train_assignment4.py check
.venv/bin/python train_assignment4.py a --session-updates 0
.venv/bin/python train_assignment4.py b --session-updates 0
.venv/bin/python train_assignment4.py status
.venv/bin/python train_assignment4.py tensorboard
.venv/bin/python train_assignment4.py drive-a
.venv/bin/python train_assignment4.py drive-b
```

명령은 해당 작업이 끝난 뒤 순서대로 실행하고 TensorBoard는 별도 터미널에서 유지한다.
CPU/CUDA 자동 선택은 기존 코드에 맡긴다. 속도를 이유로 총 학습량이나 실험 조건을 변경하지 않는다.

## 10. 사용자에게 보고할 형식

실측 결과가 생긴 뒤 다음 항목을 간단히 보고한다:

- 환경: 사용한 Python 경로, SUMO 확인 결과, 실제 device.
- A: run 경로, lr=0.00005, 목표 200000, 실제 완료 스텝/상태.
- B: run 경로, lr=0.0003, 목표 200000, 실제 완료 스텝/상태.
- 실제 평가 지표의 비교와 관찰한 주행 차이. 평가하지 않았다면 그 사실을 명시한다.
- TensorBoard 접속 주소와 직접 캡처할 그래프/시계/메모 위치.
- 결과 보관 경로와 아직 남은 작업. 최종 촬영·제출은 사용자 작업임을 유지한다.
