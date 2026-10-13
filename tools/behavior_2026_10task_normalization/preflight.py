#!/usr/bin/env python3
"""Read-only host inspection; writes a report only. No download, training or simulator."""
import argparse
import datetime
import json
import platform
from pathlib import Path
import shutil
import subprocess
import numpy as np
from compute_normalization import TASK_IDS, sha256, write_json

def command(argv, cwd=None):
    try:
        r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {'returncode': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()[:500]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    repo, data, out = Path(args.repo).resolve(), Path(args.dataset_root).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = ['2026_r1pro_xr1_state_schema.json', '2026_state_contract_team_handoff.md',
             'src/behavior_xr1/data/r1pro_2026_state.py', 'src/behavior_xr1/data/target_builder.py',
             'src/behavior_xr1/adapters/action_bridge.py',
             'artifacts/2026_state_validation/task0_reset_observation.npz']
    file_report = {f:{'exists':(repo/f).is_file(), 'sha256':sha256(repo/f) if (repo/f).is_file() else None} for f in files}
    meta = {}
    for f in ['info.json', 'tasks.jsonl', 'tasks.parquet']:
        path = data/'meta'/f
        meta[f] = {'exists':path.is_file(), 'bytes':path.stat().st_size if path.is_file() else 0}
    info = json.loads((data/'meta/info.json').read_text()) if (data/'meta/info.json').is_file() else None
    revision = next((x for x in data.parts if len(x)==40 and all(c in '0123456789abcdef' for c in x)), None)
    blockers = ['action target/consumer parity needs real trajectory evidence; no production stats yet']
    if not info:
        blockers.append('2026 local metadata missing')
    elif info.get('features', {}).get('observation.state', {}).get('shape') != [61] or info.get('features', {}).get('action', {}).get('shape') != [23]:
        blockers.append('metadata does not declare the required state61/action23 schema')
    shards = list((data/'data').rglob('*.parquet')) if (data/'data').is_dir() else []
    if not shards:
        blockers.append('no local data parquet shards found')
    report = {'status':'PREFLIGHT_ONLY', 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'repo':str(repo), 'dataset_root':str(data), 'dataset_revision_hint':revision,
        'git_branch':command(['git','branch','--show-current'], repo),
        'git_commit':command(['git','rev-parse','HEAD'], repo),
        'git_worktree':command(['git','status','--short'], repo),
        'python':platform.python_version(), 'numpy':np.__version__,
        'gpu_inventory':command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader']),
        'free_bytes':shutil.disk_usage(out).free, 'files':file_report, 'metadata':meta,
        'info':info, 'local_data_shards':len(shards), 'local_data_bytes':sum(x.stat().st_size for x in shards),
        'video_accessed':False, 'blockers':blockers}
    write_json(out/'environment_manifest.json', report)
    write_json(out/'selected_tasks.json', {'official_task_ids':TASK_IDS,
        'local_index_to_task_id':{str(i):t for i,t in enumerate(TASK_IDS)},
        'task_id_to_embedding_index':{str(t):t for t in TASK_IDS},
        'embedding_mapping_status':'PROPOSED_IDENTITY_REQUIRES_CONSUMER_VERIFICATION',
        'untrained_90_embeddings':'do not remap; retain model values; performance unverified'})
    print(json.dumps({'status':report['status'], 'local_data_shards':len(shards), 'blockers':blockers}, indent=2))

if __name__ == '__main__':
    main()
