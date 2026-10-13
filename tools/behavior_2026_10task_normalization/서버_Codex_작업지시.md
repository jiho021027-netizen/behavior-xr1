# 실제 데이터 서버의 Codex에 붙여넣을 지시

첨부 패키지 폴더를 기준으로 작업하십시오. 이번 요청의 범위는 BEHAVIOR 2026
선택 10개 태스크의 전처리와 normalization입니다. 훈련은 normalization
최종 인수 후 별도 진행하며 평가 runner/제출은 변경하지 마십시오.

1. 현재 작업 위치와 AGENTS.md를 읽고 실제 데이터 호스트인지 확인하십시오.
   laptop 제한이 적용되는 호스트라면 대용량 다운로드·시뮬레이터·모델·학습을
   실행하지 말고 데이터 호스트에 필요한 작업을 보고하십시오.
2. 공식 ID `[0,69,60,57,56,62,59,6,8,10]` 순서를 보존하십시오. 선택 10개를
   local 0..9로 바꾸더라도 100-task embedding index와 혼동하지 마십시오.
3. 현재 repo branch/commit, dataset 로컬 경로와 HF revision을 찾아
   preflight.py를 실행하십시오. N2·.venv-s2·검증된 Action Bridge는 수정 금지입니다.
4. latest training adapter와 evaluator bridge의 action 계약을 먼저 읽으십시오.
   공개 target_builder의 trunk delta, 누락 gripper, base mask가 업무 문서와
   불일치합니다. 팀 문서의 최신 구현이 서버에 있는지 확인하십시오.
   raw state61/raw action23에서 XR-1 target30×60을 만드는 검증된 코드가 없으면
   BLOCKED_ACTION_TARGET로 중단하고 필요한 정확한 파일/좌표계/단위/정렬을 보고하십시오.
5. meta task/episode/shard를 대조하여 선택 task의 실제 train episode ID 목록을
   만들고 held-out/evaluation은 배제하십시오. 전체 3.27TB 다운로드 금지입니다.
   통계에는 숫자 parquet와 meta만 필요합니다. 이미 있는 데이터를 우선 사용하십시오.
6. README에 따라 export_parquet.py로 raw를 내보내고, metadata length와 실제
   frame 수·task 이름/ID·누락을 확인하십시오. pyarrow export는 이 패키지 제작
   환경에서 real runtime 검증하지 않았으므로 Task 0 샘플부터 테스트하십시오.
7. 실제 공유 state adapter의 hash와 동봉 사본이 일치하는지 확인하십시오.
   최신 공유 schema가 다르면 팀의 양쪽 소비자를 함께 확인하십시오.
8. 검증된 builder를 build_targets API로 연결하십시오. 실측 transition에서
   EEF frame/rotation/FK/control goal/gripper/trunk/time offset을 수치로 검증하고
   실제 RGB/언어/task alignment 증거도 만드십시오. 의미 없는 padding 금지입니다.
9. compute_normalization.py로 provisional 통계와 count를 구한 뒤 전체 train
   subset, mask/tail, direct 작은 샘플 대비 streaming 통계, 소비자 공식 정규화
   식/zero-variance/reserved 정책을 검증하십시오. evidence.json에 실제 로그와
   hash를 연결하고 final 폴더로 다시 계산하십시오. shape만으로 PASS 판정 금지입니다.
10. 최종 JSON/validation/manifest/schema/commit·revision·hash를 즉시 공유하고
    소요시간, frame/episode 수, 누락/손상, horizon29 count와 remaining blockers를
    보고하십시오. 실제 데이터나 계약이 없으면 production_ready=true로 만들지 마십시오.

첫 답변은 계획만 하지 말고 preflight를 실행해 실제 repo/data 경로, commit,
존재하는 action builder와 현재 blocker를 구체적으로 보고하십시오.
