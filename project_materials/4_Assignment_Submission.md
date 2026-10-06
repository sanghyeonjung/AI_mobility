# 4차 과제 실행 및 제출 안내

`4_Model_Training_RL.md`의 PPO 실습을 바탕으로 준비한 안내입니다.
실제 모델 학습과 제출 사진 촬영은 아직 진행하지 않았습니다.

## 1. 과제 요구사항과 준비 상태

사진의 공지 기준 마감은 **10월 12일 23:59**입니다.
하이퍼파라미터를 바꿔 신규 RL 모델을 두 개 이상 학습하고, 실제 TensorBoard 그래프를 캡처합니다.
학습 시작 시각은 수업 이후여야 하며, 캡처에 학과·학번·이름과 컴퓨터 시간을 함께 표시합니다.
학습된 자율주행 차량을 SUMO 도로에 배치해 주행도 확인합니다.

| 항목 | 준비 상태 |
|---|---|
| 독립적인 신규 A/B 모델 설정 | 완료: 기존 GitHub 모델을 읽지 않고 각각 무작위 초기화 |
| 실제 시작 시각과 설정 기록 | 완료: 실행할 때 생성하여 저장 |
| 짧게 나눠 학습하고 이어서 실행 | 완료: 매 업데이트 완료 후 체크포인트 저장 |
| TensorBoard 두 모델 비교 | 명령 준비 완료: 과제 전용 폴더만 표시 |
| SUMO에서 신규 모델 주행 | 명령 준비 완료: 저장된 실험 설정 사용 |
| 학습 그래프·실제 성능 | 학습 후 생성됨 |
| 개인정보 메모·컴퓨터 시간·사진 | 사용자 직접 준비 및 촬영 |

기존 `results/run_20260907_202220`과 이전 실행 `run_20261006_135323`은 이번 두 모델에 포함하지 않습니다.
새 결과는 **`results/assignment4/`**에 저장합니다.

## 2. 두 모델의 비교 조건

| 설정 | 모델 A | 모델 B |
|---|---:|---:|
| 학습률 `lr` | `0.00005` | `0.0003` |
| 고정 목표 스텝 `TOTAL_TIMESTEPS` | 200,000 | 200,000 |
| `n_steps` | 2,048 | 2,048 |
| `n_epochs` / 미니배치 | 10 / 64 | 10 / 64 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 동일 |
| 네트워크 | 256 → 256, ReLU, hybrid | 동일 |
| 초기화 seed | 0 | 0 |
| 평가 | 5 업데이트마다 5 에피소드 | 동일 |
| 도로·보상·관측·행동 | 현재 실습 설정 | 동일 |

학습률 하나만 바꾸면 결과 차이를 해석하기 쉽습니다.
A는 원본 `train.py`의 학습률 `0.0001`보다 낮은 `0.00005`, B는 더 높은 `0.0003`으로 새로 학습합니다.
따라서 두 모델 모두 원본과 다른 학습률을 사용합니다. 총 학습량과 나머지 조건은 동일합니다.
기존 가중치를 사용하지 않으며, B가 더 좋다는 보장은 없습니다.
SUMO 교통 상황은 매 에피소드 달라지므로, 두 실행만으로 학습률의 일반적인 우열을 확정할 수는 없습니다.

## 3. 실행 준비

### 다른 PC에서 새로 준비할 때

개인 GitHub 저장소 `sanghyeonjung/AI_mobility`를 clone하거나 ZIP으로 내려받아 압축을 풉니다.

```powershell
git clone https://github.com/sanghyeonjung/AI_mobility.git
cd .\AI_mobility
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-assignment4.txt
```

그 PC에도 Python과 SUMO가 설치되어 있어야 합니다.
SUMO 설치는 [Windows 설치 안내](./1_SUMO_Windows_Setup.md)를 따릅니다.
SUMO 인스톨러의 `Set SUMO_HOME` 옵션을 선택하고 새 터미널에서 진행합니다.
가상환경 활성화 없이 `.cmd`가 프로젝트의 `.venv` Python을 사용합니다.
`python` 명령이 없다면 먼저 Python을 설치한 뒤 새 터미널을 엽니다.

```powershell
.\assignment4.cmd check
.\assignment4.cmd a
.\assignment4.cmd b
```

macOS/Linux에서는 해당 플랫폼에 SUMO를 설치한 뒤 다음처럼 실행할 수 있습니다.
macOS 설정은 [macOS 설치 안내](./1_SUMO_macOS_AppleSilicon_Setup.md)를 참고합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-assignment4.txt
.venv/bin/python train_assignment4.py check
.venv/bin/python train_assignment4.py a
.venv/bin/python train_assignment4.py b
.venv/bin/python train_assignment4.py tensorboard
```

학습 결과는 GitHub로 자동 업로드되지 않습니다.
학습한 PC에서 직접 캡처하거나, `results/assignment4/` 폴더 전체를 복사해 그래프와 모델을 옮깁니다.
TensorBoard 그래프에는 `model.pt`뿐 아니라 `tb/` 로그가 필요합니다.
수업 이후에 신규 모델 두 개를 실제로 학습해야 한다는 조건은 다른 PC에서도 같습니다.

### 현재 PC에서 실행할 때

VS Code의 PowerShell 터미널에서 프로젝트 최상위 폴더로 이동합니다.
현재 위치가 `10_6_after_mobility`라면:

```powershell
cd .\Artificial-Intelligence-and-Control-for-Autonomous-Driving
.\assignment4.cmd check
```

`OK: dependencies ready. No training started.`가 나오면 준비된 상태입니다.
이 컴퓨터의 기본 `python`은 패키지가 없는 MSYS Python을 가리킵니다.
`.cmd`는 확인된 Windows Python을 사용하며 PowerShell 실행 정책을 바꿀 필요가 없습니다.
다른 가상환경을 쓰려면 해당 Python을 지정합니다:

```powershell
$env:ASSIGNMENT_PYTHON = '본인의 학습용 python.exe 절대경로'
```

## 4. 시간이 날 때 두 모델 학습하기

### 기본: 한 번에 10 업데이트씩

```powershell
.\assignment4.cmd a
.\assignment4.cmd b
```

각 명령은 **최대 10 업데이트(20,480 스텝)** 후 자동 저장하고 종료합니다.
A가 종료된 뒤 B를 실행합니다. 시간 제한은 분 단위가 아니라 업데이트 횟수 기준입니다.
컴퓨터를 끈 뒤 **같은 명령을 다시 실행하면 각 모델의 마지막 완료 업데이트부터 이어갑니다.**
고정 목표 200,000은 rollout 단위로 올림하여 실제 200,704 스텝, 98 업데이트입니다.
각 모델을 10 업데이트씩 실행하면 약 10회 호출로 완료합니다.
이미 목표에 도달한 모델에 같은 명령을 실행하면 추가 학습하지 않습니다.

더 짧게 실행하려면:

```powershell
.\assignment4.cmd a --session-updates 1
.\assignment4.cmd b --session-updates 1
```

한 번에 끝까지 실행하려면:

```powershell
.\assignment4.cmd a --session-updates 0
.\assignment4.cmd b --session-updates 0
```

진행 상황은 `.\assignment4.cmd status`로 확인합니다.
콘솔의 예상 남은 시간은 해당 세션에서 측정한 속도 기준이며 평가 주기에 따라 달라집니다.

### 총 학습량은 변경하지 않기

사용자의 추가 수업 안내에 따라 `TOTAL_TIMESTEPS`는 두 모델 모두 **200,000으로 고정**합니다.
`--timesteps` 옵션과 이전의 5만 스텝 안내는 제거했습니다. 학습률만 변경합니다.
`--session-updates`는 한 번 실행할 때의 작업량이며, 모든 세션을 합친 최종 학습량은 바뀌지 않습니다.
시간이 부족하면 세션을 나눠 실행하거나 다른 PC에서 학습합니다.

### 이전 설정으로 A를 이미 학습했다면

이전 학습률 `0.0001`의 A는 이번 최종 설정의 A로 포함하지 않습니다.
기존 결과를 보존하고 `.\assignment4.cmd a --new-run`으로 `0.00005`의 A를 처음부터 학습합니다.
기존 A 체크포인트의 학습률만 바꿔 이어 학습하거나 기록을 수정하지 않습니다.
B가 이미 `0.0003`, 200,000 목표로 실행 중이면 해당 설정을 확인한 뒤 기존 run을 완료해도 됩니다.
실행 중에는 코드를 업데이트하지 않으며, 코드 변경 감지 오류는 인계서의 기존 실행 확인 절차를 따릅니다.

### Ctrl+C와 재개

Ctrl+C를 한 번 누르고 `Saved:`가 나올 때까지 기다립니다.
정책·Adam optimizer·누적 스텝·난수 상태가 담긴 `checkpoint.pt`에서 재개합니다.
업데이트 도중 중단하면 미완료 수집/수정은 버리고 마지막 완료 업데이트로 돌아갑니다.
첫 업데이트 전 중단했다면 학습 스텝은 0이므로 학습 완료 모델로 볼 수 없습니다.
강제로 전원을 끄면 마지막으로 성공적으로 저장된 체크포인트까지만 복구됩니다.

SUMO의 진행 중인 에피소드와 최근 20 에피소드 통계는 재개 시 새로 시작합니다.
따라서 무중단 실행과 완전히 같은 궤적은 아닙니다. 같은 run 이름과 최초 시작 시각은 유지됩니다.
학습을 시작한 뒤 관련 코드/설정을 바꾸면 기존 run 재개를 차단합니다.
별도 새 실험을 만들 때만 `--new-run`을 붙이고, 이번 두 모델만 비교할 때는 붙이지 않습니다.

## 5. TensorBoard에서 실제 그래프 보기

다른 터미널을 열고 같은 프로젝트 폴더에서:

```powershell
.\assignment4.cmd tensorboard
```

브라우저에서 `http://localhost:6006`에 접속합니다.
`results/assignment4/`만 읽으므로 GitHub의 기존 모델이 포함되지 않습니다.

1. Scalars 또는 Time Series에서 `model_a_실제날짜_시각/tb`, `model_b_실제날짜_시각/tb`를 모두 선택합니다.
2. 공지 예시와 같은 `episode/return`, `episode/length`를 표시합니다.
3. `train/ep_rew_mean`, `eval/ep_return`도 확인합니다.
4. `eval/collision_rate`, `eval/success_rate`로 실제 주행 성능을 비교합니다.
5. x축은 `Step`, smoothing은 두 모델에 동일한 값을 사용합니다.
6. 왼쪽 run 이름에 실제 학습 날짜·시간이 보이게 캡처합니다.

Text 탭의 `experiment/settings`에 실제 설정이, `experiment/started_at_kst`에 최초 실행 시각이 표시됩니다.
시간은 KST 기준 실제 컴퓨터 시각으로 기록합니다. 수업 이후에 실제 신규 학습을 시작해야 합니다.
수업 시간을 맞추기 위해 run 이름이나 로그 시각을 임의로 바꾸지 않습니다.

학습 전에는 제출용 그래프가 없습니다. `No dashboards are active`가 뜨면 실제 학습 시작 여부와 로그 경로를 확인합니다.
6006 포트를 이전 TensorBoard가 쓰고 있다면 그 서버를 종료한 뒤 위 명령을 실행합니다.

## 6. 학습된 차량의 도로 주행 확인

```powershell
.\assignment4.cmd drive-a
.\assignment4.cmd drive-b
```

각 명령은 해당 모델의 최신 과제 run을 찾아 1 에피소드를 GUI로 재생합니다.
SUMO 창의 **▶ 버튼**을 눌러 시작합니다. 빨간 ego 차량이 학습된 정책으로 주행합니다.
두 모델은 순서대로 실행하고, 필요한 장면을 직접 캡처합니다.

화면 없이 5 에피소드 평가가 필요하면 실제 run 경로를 넣습니다:

```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" .\test.py .\results\assignment4\실제_run폴더\model.pt --algorithm ppo --episodes 5 --nogui
```

과제 모델은 저장된 `experiment_config.json`의 도로·보상·네트워크 설정을 사용합니다.
출력되는 충돌률·완주율·평균 속도·차선변경·평균 보상을 비교합니다.
보상 상승뿐 아니라 충돌 감소와 완주율 상승을 함께 확인하세요. 실제 결과가 생기기 전에는 우열을 작성하지 않습니다.

## 7. 사용자가 직접 촬영할 제출 사진

- **필수 그래프 사진:** 새 A/B run 이름과 두 모델의 실제 TensorBoard 그래프가 보이도록 배치합니다.
- **개인정보:** 메모장 등에 학과·학번·이름을 작성하여 그래프와 함께 표시합니다.
- **컴퓨터 시간:** 작업 표시줄 시계를 포함합니다. 시간이 안 보이면 공지처럼 별도 시계를 띄웁니다.
- **도로 주행 사진:** SUMO에서 학습된 차량의 주행 장면을 추가로 캡처하면 도로 배치 요구도 확인할 수 있습니다.
- JPG, JPEG, PNG, PDF 중 허용 형식으로 제출합니다.

공지에서 명시한 필수 캡처는 TensorBoard 그래프입니다.
SUMO 주행 사진은 도로 배치 요구를 확인하기 위한 보완 사진으로 권장합니다.
개인정보 메모와 실제 사진은 사용자가 직접 준비합니다.

## 8. 결과 파일

```text
results/assignment4/
  model_a_YYYYMMDD_HHMMSS_ffffff/
    model.pt                 # 주행용 정책 가중치
    checkpoint.pt            # 학습 재개용 상태
    experiment_config.json   # 실제 실험 설정
    run_info.json            # 시작 시각, 세션 기록, 스텝, 상태
    training_log.csv
    eval_log.csv             # 평가 실행 후 생성
    tb/                      # 실제 학습 TensorBoard 로그
    source/                  # 시작 당시 관련 코드 사본
    sumo/                    # 해당 run 전용 도로
  model_b_YYYYMMDD_HHMMSS_ffffff/
    ...
```

준비 점검은 임시 폴더에서 진행하며 제출용 run이나 학습 그래프를 만들지 않습니다.
