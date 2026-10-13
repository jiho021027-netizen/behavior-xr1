# BEHAVIOR 2026 — 10개 태스크 정규화 실행 패키지

대상: `[0,69,60,57,56,62,59,6,8,10]`.

작성한 것은 실제 데이터 서버에서 실행할 전처리·통계 계산 코드입니다.
실제 2026 데이터를 읽거나 최종 통계 JSON을 계산한 결과가 아닙니다.
GPU 학습·평가 runner·제출은 실행하지 않았습니다. 공개 GitHub 브랜치와
업로드된 10/12 업무 문서를 기준으로 만들었습니다.

## 현재 먼저 해결할 사항

현재 공개 코드의 action target 정의가 업무 문서의 정의와 일치하지 않습니다.
`public_code_audit.json`에 조회한 정확한 commit과 파일·함수를 기록했습니다.

| 공개 코드 | 확인한 문제 |
|---|---|
| `data/target_builder.py` | trunk `[20:24]`를 미래 qpos−현재 qpos로 만들지만 업무 문서는 absolute position을 요구합니다. 그리퍼 `[6,14]`는 마스크에서 빠져 있고 base `[17:20]`는 포함되어, 업무 문서의 활성 18채널과도 다릅니다. |
| `data/behavior_r1pro.py::convert_native_episode` | 2026 61D가 아니라 256D를 요구하고, 이 브랜치에 없는 모듈을 가져옵니다. |
| `adapters/action_bridge.py` | 공개 버전은 trunk 고정·gripper hold입니다. 팀 문서의 최신 bridge와 동일 코드라는 증거가 없습니다. |

따라서 이 패키지는 기존 builder를 임의로 재사용하지 않습니다. 팀의 **실제 최신
23D→30×60 target builder와 평가 bridge를 같은 정의로 확정**한 후 연결합니다.
관측 EEF 변화량은 실제 움직임이고 raw action은 제어 목표이므로, 둘을 자동으로
같다고 취급할 수 없습니다. gripper positive=OPEN이라는 검증도 그 자체로
XR-1 gripper delta의 물리 단위·스케일을 확정하지는 않습니다.

## 들어 있는 코드와 검증 범위

- `preflight.py`: 로컬 repo commit/파일 hash/메타데이터/디스크/GPU 목록을 조회합니다.
- `export_parquet.py`: 이미 내려받은 parquet에서 선정된 학습 episode의 숫자 열만
  읽어 episode별 raw NPZ로 내보냅니다. RGB 비디오는 읽지 않습니다.
- `prepare_episodes.py`: 동봉한 공유 state adapter를 그대로 사용하고, 팀이 검증한
  action builder 플러그인을 호출하여 episode별 dense NPY와 manifest를 만듭니다.
- `compute_normalization.py`: 정확한 state 분위수와 mask를 적용한 action mean/std를
  계산합니다. 실제 샘플 검증 증거가 없으면 `production_ready=false`입니다.
- `reference/`: 조회 시점의 공유 61D→60D state adapter와 schema 사본입니다.
- `tests/`: 작은 합성 데이터 기반 CPU 테스트입니다. 실제 데이터 검증이 아닙니다.

통계 계산, mask와 episode 경계, 중복/잘못된 task 거부, 공유 state mapping을
합성 데이터로 테스트한 5개 검사가 통과했습니다. parquet export는 컴파일만
확인했으며 이 환경에는 pyarrow 및 실제 parquet 데이터가 없어 실행 검증하지
못했습니다. 실행 후 Task 0 실제 샘플 결과를 확인해야 합니다.

## 1. 서버에서 첫 실행

ZIP을 풀고 이 README가 있는 폴더에서 실행합니다. 아래 두 경로는 실제 서버에
맞게 바꾸십시오. 공개 repo와 별도 폴더에서 실행할 수 있으며 기존 파일을
덮어쓰거나 N2·`.venv-s2`·Action Bridge를 변경하지 않습니다.

```bash
python preflight.py \
  --repo /실제/behavior-xr1 \
  --dataset-root /실제/2026-challenge-demos \
  --output-dir ./outputs/preflight
```

첫 인수물: `outputs/preflight/environment_manifest.json`, `selected_tasks.json`.
지금 데이터 경로가 없으면 이 보고서에서 그 사실을 확인하고 데이터 호스트에서
선정 task의 meta·숫자 parquet만 먼저 확보합니다. 이 패키지는 다운로드를 하지
않으며 전체 데이터/모델/시뮬레이터를 설치하지 않습니다.

## 2. 정확한 학습 subset만 raw export

호스트에 NumPy와 pyarrow가 필요합니다. `train_episode_ids.json`은 팀이 실제
학습에 사용하는 선택 10-task episode ID의 JSON 배열입니다. meta의 episode
인덱스와 shard 매핑, 실제 train/held-out 분할을 확인해서 만드십시오. 이 패키지가
train split을 임의로 만들어 주지는 않습니다. 평가 instance 데이터는 넣지 않습니다.

`task_map.json` 형식은 다음과 같습니다. 아래 identity mapping은 **검증 전 예시**입니다.
공식 task ID와 dataset task_index가 같은지 `meta/tasks`에서 이름까지 대조한 뒤
상태를 `VERIFIED_BY_TEAM`으로 바꾸십시오. task ID를 local 0..9로 바꾸지 마십시오.

```json
{
  "status": "PENDING",
  "official_task_id_to_dataset_task_index": {
    "0": 0, "69": 69, "60": 60, "57": 57, "56": 56,
    "62": 62, "59": 59, "6": 6, "8": 8, "10": 10
  }
}
```

```bash
python export_parquet.py \
  --dataset-root /실제/2026-challenge-demos \
  --dataset-revision 실제_HF_40자리_commit_SHA \
  --train-episode-ids ./train_episode_ids.json \
  --task-map ./task_map.json \
  --output-dir ./outputs/raw
```

로컬에 없는 shard의 episode는 `missing_episodes`, 잘못된 row는 `damaged_episodes`
에 기록합니다. 원본 shard 경로도 manifest에 남깁니다. export 결과가 원본
episode 메타데이터의 length와 일치하는지 별도로 확인하십시오. parquet에서
task_index 등 필요한 열을 찾지 못하면 추정하지 않고 중단합니다.

## 3. 검증된 action builder 연결

팀이 승인한 builder 파일은 다음 함수를 제공해야 합니다. 프로젝트 import가
필요하다면 프로젝트의 정상 환경에서 실행하고 PYTHONPATH를 설정하십시오.

```python
def build_targets(raw_states, raw_actions, frame_index, timestamp, *, horizon=30):
    # raw_states: float32 (N,61), raw_actions: float32 (N,23)
    # returns: targets float32 (N,30,60), masks bool (N,30,60)
    # 실제 검증한 EEF/controller/FK/frame/gripper/trunk 정의로 구현
    # h의 future offset은 현재 관측에서 h+1; 이 규약 자체도 팀 검증 필요
    # 마지막 episode 밖의 timestep은 false; 활성채널은 문서의 18개
    ...
```

`action_schema.json`에는 최소한 다음과 더불어 좌표계, 단위, gripper delta/scale,
trunk 절대 위치, source action과 관측의 시간 정렬, base 처리, 평가 소비자
파일/commit 및 실제 transition 검증 근거를 기록하십시오.

```json
{
  "status": "PENDING",
  "horizon": 30,
  "target_offset": "h+1_from_current_observation",
  "active_action_indices": [0,1,2,3,4,5,6,8,9,10,11,12,13,14,20,21,22,23],
  "target_builder_sha256": "실제 검증한 builder 파일 SHA256"
}
```

의미 검증을 완료한 후 `VERIFIED_BY_TEAM`으로 표시합니다. `raw23D`를 패딩해서
출력 shape만 맞추면 안 됩니다. latest training/evaluation에서 horizon 또는
활성 채널 정의가 다르면 이번 패키지에도 동일하게 반영한 뒤 테스트하십시오.

```bash
python prepare_episodes.py \
  --raw-manifest ./outputs/raw/raw_manifest.json \
  --builder /실제/검증된_action_builder.py \
  --action-schema ./action_schema.json \
  --output-dir ./outputs/converted
```

형상·finite·frame/timestamp·mask를 검사하지만 RGB+언어 정렬 및 물리 의미를
자동으로 PASS로 만들지는 않습니다. `task0_real_batch_validation.json`에도
이 제한을 명시합니다. tail의 값은 0/반복값 중 무엇이든 가능하나 mask는 false입니다.

## 4. 통계 계산

```bash
python compute_normalization.py \
  --manifest ./outputs/converted/train_manifest.json \
  --output-dir ./outputs/stats_provisional \
  --scratch-dir /충분한_공간이_있는_임시경로
```

출력은 `2026_10task_normalization.json`과 `normalization_validation.json`입니다.
**최종 증거가 없으므로 위 명령은 provisional 결과를 만들고 exit code 2를 반환합니다.**
실패는 1, 증거를 포함한 최종 통계는 0입니다. 새 출력 폴더를 사용하십시오.

| JSON key | shape | 방법 |
|---|---|---|
| `state_q01`, `state_q99` | `(1,60)` | 모든 학습 frame을 1회씩, 채널별 정확한 1%/99% 분위수 |
| `action_mean`, `action_std` | `(30,60)` | 각 anchor의 유효 미래 timestep만, horizon·채널별 population 통계 |
| `action_valid_count` | `(30,60)` | horizon별 표본 수; episode tail은 제외 |

action은 float64 Welford, std는 ddof=0입니다. state quantile은 float64 보간으로
정확히 계산하며 20개 활성 state 채널을 디스크에 임시 저장합니다. 추가 디스크는
약 `N×20×4 byte`이고, quantile 계산은 한 채널의 N개 값을 메모리에 올립니다.
변환 target NPY 자체는 `N×30×60×4 byte`이므로 여유 공간을 확인하십시오.
RGB를 읽지 않으며 저장된 target을 다시 계산하지 않습니다.

reserved 채널의 raw state는 0, action mask는 false, 통계는 0입니다.
상수 활성 action의 std도 실제 값 0을 저장합니다. epsilon을 std 자체에 미리
더하지 않습니다. 소비자가 `std+1e-6`으로 정규화하는지 팀 consumer를 확인하십시오.
state zero-range의 normalized 값과 reserved mask 적용도 consumer parity 검사에
포함하십시오. 통계 JSON만 만들었다고 학습/평가 적용이 완료된 것은 아닙니다.

## 5. 최종 증거 게이트

실제 샘플 검증 후 `evidence.json`을 작성합니다. 다음 항목의 PASS 상태와 실제
로그 경로/hash가 모두 필요합니다. 먼저 계산한 provisional validation에서
`converted_records_sha256`을 가져와 동일한 배열에 증거를 연결할 수 있습니다.

```json
{
  "train_manifest_sha256": "train_manifest.json hash",
  "schema_and_builder_sha256": {
    "state_schema": "hash", "action_schema": "hash", "target_builder": "hash"
  },
  "converted_records_sha256": "provisional validation의 값",
  "checks": {
    "state_mapping": {"status":"PENDING", "log_path":"", "log_sha256":""},
    "action_target_semantics": {"status":"PENDING", "log_path":"", "log_sha256":""},
    "rgb_language_task_alignment": {"status":"PENDING", "log_path":"", "log_sha256":""},
    "episode_boundaries": {"status":"PENDING", "log_path":"", "log_sha256":""},
    "train_subset_complete": {"status":"PENDING", "log_path":"", "log_sha256":""},
    "consumer_contract_parity": {"status":"PENDING", "log_path":"", "log_sha256":""}
  }
}
```

이 게이트는 팀의 실제 검증 결과를 입력받아 로그와 배열 hash를 확인하는 방식입니다.
로그 내용의 진실성이나 물리적 정확성 자체를 자동으로 증명하는 장치는 아닙니다.
근거 없이 상태만 PASS로 고치면 안 됩니다.

```bash
python compute_normalization.py \
  --manifest ./outputs/converted/train_manifest.json \
  --evidence ./evidence.json \
  --output-dir ./outputs/stats_final
```

선택 train episode 전체 포함, 모든 task 포함, 누락/손상 0, 유효 표본 수,
schema/builder/로그 hash, consumer parity가 충족되어야 최종 상태가 true입니다.
샘플러가 task 균형을 사용하는 경우도 이 통계의 모집단은 **frame별 비가중**이라는
점을 명시하고, 팀에서 다른 모집단을 요구하면 통계 정의부터 합의하십시오.

평가 consumer가 flat key를 요구하므로 문서의 `state.q01` 표기는 JSON에서는
`state_q01`로 저장합니다. 기존 공개 consumer는 shape만 검사하므로 추가로
production_ready와 schema/hash 일치를 반드시 확인해야 합니다.

## 가장 먼저 팀에 전달할 것

최종 normalization, validation, train_manifest, state/action schema, builder
commit/hash와 Task 0 실제 검증 로그입니다. 학습 체크포인트를 기다릴 필요는 없습니다.
현재 이 패키지 자체의 상태는 `IMPLEMENTED / CPU_SYNTHETIC_TEST_PASS`이고
`REAL_DATA_T2_T3=NOT_EXECUTED`, `NORMALIZATION_PRODUCTION_READY=NO`입니다.
