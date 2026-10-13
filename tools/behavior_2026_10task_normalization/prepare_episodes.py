#!/usr/bin/env python3
"""Convert local per-episode NPZ exports using a team's verified action builder.

No dataset/video/model downloads. Refuse to silently use old raw256D code.
The action builder is an explicit Python plugin, never chosen automatically.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from compute_normalization import TASK_IDS, sha256, write_json, require, inspect_episode
from reference.r1pro_2026_state import adapt_r1pro_2026_state

def prepare(raw_manifest_path, builder_path, action_schema_path, output_dir):
    manifest_path = Path(raw_manifest_path).resolve()
    raw = json.loads(manifest_path.read_text())
    require(raw['official_task_ids'] == TASK_IDS, 'task order mismatch')
    require(Path(builder_path).is_file(), 'BLOCKED_ACTION_TARGET: verified builder file missing')
    schema = json.loads(Path(action_schema_path).read_text())
    require(schema.get('status') == 'VERIFIED_BY_TEAM',
            'BLOCKED_ACTION_TARGET: action schema is not verified by team')
    require(schema.get('active_action_indices') == list(range(7))+list(range(8,15))+list(range(20,24)),
            'action schema must match the agreed 18 dense channels')
    require(schema.get('horizon') == 30 and schema.get('target_offset') == 'h+1_from_current_observation',
            'action horizon/offset mismatch')
    require(schema.get('target_builder_sha256') == sha256(builder_path), 'target builder hash mismatch')
    spec = importlib.util.spec_from_file_location('team_verified_action_builder', builder_path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    require(callable(getattr(plugin, 'build_targets', None)), 'builder must export build_targets(...)')
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'train_manifest.json').exists(), 'use a fresh output directory')
    # Bind builder and schemas by bytes, without copying project/vendor dependencies.
    (out / 'target_builder.py').write_bytes(Path(builder_path).read_bytes())
    (out / 'state_schema.json').write_bytes((Path(__file__).parent / 'reference/2026_r1pro_xr1_state_schema.json').read_bytes())
    (out / 'action_schema.json').write_bytes(Path(action_schema_path).read_bytes())
    records = []
    sample_report = None
    for item in raw['episodes']:
        require(item['task_id'] in TASK_IDS and item['split'] == 'train', 'nonselected or nontrain episode')
        eid = item['episode_id']
        require(isinstance(eid, int) and eid >= 0, 'episode ID must be nonnegative integer')
        dest = out / f'episode_{eid:06d}'
        dest.mkdir(exist_ok=False)
        path = (manifest_path.parent / item['raw_npz']).resolve()
        with np.load(path, allow_pickle=False) as data:
            s, a = data['observation.state'], data['action']
            frame, ts = data['frame_index'], data['timestamp']
            require(s.ndim == 2 and s.shape[1] == 61 and a.shape == (len(s), 23),
                    '2026 raw input must be (N,61) state and (N,23) action')
            require(s.dtype == a.dtype == np.dtype('float32'), 'raw state/action dtype must be float32')
            require(np.isfinite(s).all() and np.isfinite(a).all(), 'raw episode nonfinite')
            require(len(s) == item['frames'], 'raw manifest frame count mismatch')
            states = np.stack([adapt_r1pro_2026_state(row) for row in s])
            # Plugin owns FK/controller/frame/gripper/trunk/time definitions.
            # In particular raw 23D is never padded into dense 60D.
            targets, masks = plugin.build_targets(s, a, frame, ts, horizon=30)
            require(np.asarray(targets).dtype == np.dtype('float32'), 'builder targets must be float32')
            for key, value in [('states', states), ('actions', targets), ('mask', masks),
                               ('frame_index', frame), ('timestamp', ts)]:
                np.save(dest / f'{key}.npy', value, allow_pickle=False)
            record = {'task_id': item['task_id'], 'episode_id': eid, 'split': 'train',
                      'frames': len(s), 'source_shards': item['source_shards'],
                      'raw_npz_sha256': sha256(path)}
            for key in ('states', 'actions', 'mask', 'frame_index', 'timestamp'):
                record[key] = f'{dest.name}/{key}.npy'
            inspect_episode(out, record, 128)
            records.append(record)
            if item['task_id'] == 0 and sample_report is None:
                sample_report = {'status': 'STRUCTURE_CHECKED_ON_HOST_SAMPLE',
                    'episode_id': eid, 'raw_sha256': sha256(path),
                    'shapes': {'raw_state': list(s.shape), 'raw_action': list(a.shape),
                               'state': list(states.shape), 'target': list(np.shape(targets)),
                               'mask': list(np.shape(masks))},
                    'rgb_language_task_alignment': 'NOT_CHECKED_BY_THIS_NO_VIDEO_TOOL',
                    'controller_semantics': 'REQUIRES_TEAM_LOG_EVIDENCE',
                    'state_adapter_sha256': sha256(Path(__file__).parent / 'reference/r1pro_2026_state.py')}
    final = {k: raw[k] for k in ('dataset_repo', 'dataset_revision', 'official_task_ids',
             'expected_train_episode_ids', 'missing_episodes', 'damaged_episodes')}
    final.update({'episodes': records, 'state_schema': 'state_schema.json',
        'source_data_kind': raw.get('source_data_kind', 'UNKNOWN'),
        'action_schema': 'action_schema.json', 'target_builder': 'target_builder.py',
        'task_id_to_embedding_index': {str(t):t for t in TASK_IDS},
        'local_index_to_task_id': {str(i):t for i,t in enumerate(TASK_IDS)},
        'target_offset': 'h+1_from_current_observation',
        'sampling_population': 'one_state_and_one_anchor_per_frame'})
    write_json(out / 'train_manifest.json', final)
    write_json(out / 'task0_real_batch_validation.json', sample_report or {'status':'BLOCKED_TASK0_MISSING'})
    return out / 'train_manifest.json'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-manifest', required=True)
    p.add_argument('--builder', required=True)
    p.add_argument('--action-schema', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    try:
        print(prepare(args.raw_manifest, args.builder, args.action_schema, args.output_dir))
    except (ValueError, KeyError, OSError) as exc:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'prepare_failure.json', {'status':'BLOCKED_ACTION_TARGET_OR_DATA', 'reason':str(exc)})
        raise SystemExit(f'BLOCKED: {exc}')
