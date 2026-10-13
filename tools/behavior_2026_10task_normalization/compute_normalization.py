#!/usr/bin/env python3
"""CPU-only stats for verified dense XR-1 episodes; never infer action semantics.

Input: manifest of memory-mappable .npy states (N,60), targets/masks (N,30,60).
Quantiles are exact linear quantiles, using an on-disk buffer one channel at a
time. Actions use float64 parallel Welford updates with channel/horizon masks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np

TASK_IDS = [0, 69, 60, 57, 56, 62, 59, 6, 8, 10]
STATE_ACTIVE = list(range(16)) + list(range(20, 24))
ACTION_ACTIVE = list(range(7)) + list(range(8, 15)) + list(range(20, 24))
CHECKS = ['state_mapping', 'action_target_semantics', 'rgb_language_task_alignment',
          'episode_boundaries', 'train_subset_complete', 'consumer_contract_parity']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def require(condition, message):
    if not condition:
        raise ValueError(message)

class Moments:
    def __init__(self):
        self.count = np.zeros((30, 60), dtype=np.int64)
        self.mean = np.zeros((30, 60), dtype=np.float64)
        self.m2 = np.zeros((30, 60), dtype=np.float64)

    def update(self, actions, masks):
        # Invalid/padded data never contributes, including invalid NaN targets.
        a = np.where(masks, actions, 0).astype(np.float64)
        n = masks.sum(axis=0, dtype=np.int64)
        mu = np.divide(a.sum(axis=0), n, out=np.zeros_like(self.mean), where=n > 0)
        delta = np.where(masks, a - mu, 0)
        m2 = np.square(delta).sum(axis=0)
        total = self.count + n
        d = mu - self.mean
        fraction = np.divide(n, total, out=np.zeros_like(self.mean), where=total > 0)
        cross = np.divide(self.count.astype(float) * n, total,
                          out=np.zeros_like(self.mean), where=total > 0)
        self.mean += d * fraction
        self.m2 += m2 + np.square(d) * cross
        self.count = total

    def std(self):
        var = np.divide(self.m2, self.count, out=np.zeros_like(self.mean), where=self.count > 0)
        return np.sqrt(np.maximum(var, 0))  # population, ddof=0

def array(root, episode, key):
    path = (root / episode[key]).resolve()
    return path, np.load(path, mmap_mode='r', allow_pickle=False)

def inspect_episode(root, ep, block_size):
    sp, s = array(root, ep, 'states')
    ap, a = array(root, ep, 'actions')
    mp, m = array(root, ep, 'mask')
    fp, frames = array(root, ep, 'frame_index')
    tp, times = array(root, ep, 'timestamp')
    n = int(ep['frames'])
    require(n > 0 and s.shape == (n, 60), f"episode {ep['episode_id']}: states must be (N,60)")
    require(a.shape == m.shape == (n, 30, 60), 'actions/mask must be (N,30,60)')
    require(frames.shape == times.shape == (n,), 'frame_index/timestamp must be (N,)')
    require(frames.dtype.kind in 'iu' and np.array_equal(frames, np.arange(n)),
            'each episode must contain every frame exactly once, in order starting at 0')
    require(np.isfinite(times).all() and (n == 1 or (np.diff(times) > 0).all()),
            'timestamps must be finite and strictly increasing within episode')
    require(s.dtype == a.dtype == np.dtype('float32'), 'states/actions must be float32')
    require(m.dtype.kind in 'bui', 'mask must be boolean or binary integer')
    require(ep.get('split') == 'train', 'only train episodes are accepted')
    reserved_s = sorted(set(range(60)) - set(STATE_ACTIVE))
    reserved_a = sorted(set(range(60)) - set(ACTION_ACTIVE))
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sb, ab, mb = s[start:stop], a[start:stop], m[start:stop]
        require(np.isfinite(sb).all(), 'states contain non-finite values')
        require(np.isin(mb, [0, 1]).all(), 'mask is not binary')
        require(np.isfinite(ab[mb.astype(bool)]).all(), 'valid targets contain non-finite values')
        require((sb[:, reserved_s] == 0).all(), 'reserved states must be zero before normalization')
        require((mb[:, :, reserved_a] == 0).all(), 'reserved actions must have false mask')
        # Target offset k=h+1. No future states may come from the next episode.
        expected = (np.arange(start, stop)[:, None] + np.arange(1, 31)[None, :] < n)
        require(np.array_equal(mb[:, :, ACTION_ACTIVE].astype(bool),
                               np.broadcast_to(expected[:, :, None], (stop-start, 30, 18))),
                'active mask must match future h+1 offsets and episode tail exactly')
    hashes = {key: sha256(p) for key, p in
              [('states', sp), ('actions', ap), ('mask', mp), ('frame_index', fp), ('timestamp', tp)]}
    return s, a, m, hashes

def compute(manifest_path, output_dir, *, block_size=128, evidence_path=None, scratch_dir=None):
    start_time = time.monotonic()
    manifest_path, out = Path(manifest_path).resolve(), Path(output_dir).resolve()
    require(block_size > 0, 'block-size must be positive')
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / '2026_10task_normalization.json').exists(),
            'output already contains normalization; use a fresh output directory')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['official_task_ids'] == TASK_IDS, 'fixed official task order mismatch')
    require(manifest['dataset_repo'] == 'behavior-1k/2026-challenge-demos', 'wrong dataset')
    revision = manifest.get('dataset_revision', '')
    require(len(revision) == 40 and all(c in '0123456789abcdef' for c in revision),
            'dataset revision must be a pinned 40-character commit SHA')
    require(manifest.get('task_id_to_embedding_index') == {str(t): t for t in TASK_IDS},
            'this package uses official IDs as the 100-task embedding indices')
    require(manifest.get('target_offset') == 'h+1_from_current_observation',
            'target timestep definition is missing or unsupported')
    episodes = manifest['episodes']
    require(bool(episodes), 'no converted episodes supplied')
    pairs = [(e['task_id'], e['episode_id']) for e in episodes]
    require(len(set(e['episode_id'] for e in episodes)) == len(episodes), 'duplicate global episode ID')
    require(all(t in TASK_IDS for t, _ in pairs), 'nonselected task in manifest')
    require(set(t for t, _ in pairs) == set(TASK_IDS), 'at least one selected task is missing')
    for key in ('state_schema', 'action_schema', 'target_builder'):
        require(bool(manifest.get(key)), f'{key} path is required')
        require((manifest_path.parent / manifest[key]).is_file(), f'{key} file missing')
    schema_hashes = {key: sha256(manifest_path.parent / manifest[key])
                     for key in ('state_schema', 'action_schema', 'target_builder')}
    total_frames = sum(int(e['frames']) for e in episodes)
    require(total_frames > 0, 'empty dataset')
    moments = Moments()
    records = []
    # 20 active columns x N x 4 bytes; exact quantiles do not need RGB or full actions in RAM.
    if scratch_dir:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='xr1-stats-', dir=scratch_dir) as tmp:
        state_buffer = np.lib.format.open_memmap(Path(tmp) / 'states.npy', mode='w+',
                                                 dtype='float32', shape=(total_frames, len(STATE_ACTIVE)))
        cursor = 0
        for ep in episodes:
            s, a, m, hashes = inspect_episode(manifest_path.parent, ep, block_size)
            n = len(s)
            for b in range(0, n, block_size):
                end = min(b + block_size, n)
                state_buffer[cursor+b:cursor+end] = s[b:end, STATE_ACTIVE]
                moments.update(a[b:end], m[b:end].astype(bool))
            records.append({'task_id': ep['task_id'], 'episode_id': ep['episode_id'],
                            'frames': n, 'sha256': hashes, 'source_shards': ep.get('source_shards', [])})
            cursor += n
        state_buffer.flush()
        q01, q99 = np.zeros((1, 60)), np.zeros((1, 60))
        for col, channel in enumerate(STATE_ACTIVE):
            q01[0, channel], q99[0, channel] = np.quantile(np.asarray(state_buffer[:, col], dtype=np.float64), [0.01, 0.99])
        del state_buffer
    std = moments.std()
    require((moments.count[:, ACTION_ACTIVE] > 0).all(), 'some horizon/channel has no valid sample')
    stats = {'state_q01': q01.tolist(), 'state_q99': q99.tolist(),
             'action_mean': moments.mean.tolist(), 'action_std': std.tolist()}
    blockers = []
    if manifest.get('source_data_kind') != 'real_2026_demo':
        blockers.append('input provenance is not declared as real_2026_demo')
    evidence = json.loads(Path(evidence_path).read_text()) if evidence_path else {}
    if evidence.get('train_manifest_sha256') != sha256(manifest_path):
        blockers.append('evidence is absent or does not bind this exact train manifest')
    if evidence.get('schema_and_builder_sha256') != schema_hashes:
        blockers.append('state/action schema and target builder hashes are not approved')
    actual_records_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if evidence.get('converted_records_sha256') != actual_records_hash:
        blockers.append('converted arrays are not bound to real-sample validation evidence')
    for check in CHECKS:
        record = evidence.get('checks', {}).get(check, {})
        if record.get('status') != 'PASS' or not record.get('log_path') or not record.get('log_sha256'):
            blockers.append(f'{check}: real-sample/consumer evidence is missing')
        else:
            log = (Path(evidence_path).resolve().parent / record['log_path']).resolve()
            if not log.is_file() or sha256(log) != record['log_sha256']:
                blockers.append(f'{check}: evidence log missing or hash mismatch')
    expected = manifest.get('expected_train_episode_ids')
    if not isinstance(expected, list) or len(set(expected)) != len(expected) or set(expected) != set(e['episode_id'] for e in episodes):
        blockers.append('exact coverage of independently selected train episode inventory is unverified')
    if manifest.get('missing_episodes') != [] or manifest.get('damaged_episodes') != []:
        blockers.append('missing/damaged episodes unresolved')
    if manifest.get('sampling_population') != 'one_state_and_one_anchor_per_frame':
        blockers.append('sampling population is not the supported unweighted frame population')
    stats.update({'production_ready': not blockers, 'final_stats_ready': not blockers,
                  'dataset_repo': manifest['dataset_repo'], 'dataset_revision': revision,
                  'official_task_ids': TASK_IDS, 'action_horizon': 30,
                  'state_active_indices': STATE_ACTIVE, 'action_active_indices': ACTION_ACTIVE,
                  'action_valid_count': moments.count.tolist(), 'state_frame_count': total_frames,
                  'statistics': {'action': 'masked_population_mean_std_ddof0_per_horizon',
                                 'state': 'exact_linear_quantiles', 'weighting': 'one_anchor_per_frame'},
                  'train_manifest_sha256': sha256(manifest_path),
                  'schema_and_builder_sha256': schema_hashes,
                  'converted_records_sha256': actual_records_hash,
                  'epsilon': 1e-6, 'blockers': blockers})
    validation = {'status': 'COMPUTED_ON_SUPPLIED_ARRAYS', 'production_ready': not blockers,
                  'runtime_seconds': time.monotonic() - start_time, 'episodes': len(episodes),
                  'frames': total_frames, 'video_accessed': False,
                  'source_data_kind': manifest.get('source_data_kind', 'UNKNOWN'),
                  'shapes': {k: list(np.asarray(v).shape) for k, v in stats.items()
                             if k in ('state_q01', 'state_q99', 'action_mean', 'action_std')},
                  'schema_and_builder_sha256': schema_hashes, 'records': records,
                  'converted_records_sha256': actual_records_hash,
                  'zero_variance_action': np.argwhere((std == 0) & (moments.count > 0)).tolist(),
                  'zero_range_state': [i for i in STATE_ACTIVE if q01[0, i] == q99[0, i]],
                  'normalization_consumer': 'flat keys match public run_zero_shot.load_normalization',
                  'consumer_formula_verified': not any('consumer_contract_parity' in b for b in blockers),
                  'blockers': blockers}
    write_json(out / '2026_10task_normalization.json', stats)
    write_json(out / 'normalization_validation.json', validation)
    return stats, validation

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--evidence')
    p.add_argument('--block-size', type=int, default=128)
    p.add_argument('--scratch-dir')
    args = p.parse_args()
    try:
        stats, report = compute(args.manifest, args.output_dir, block_size=args.block_size,
                                evidence_path=args.evidence, scratch_dir=args.scratch_dir)
        print(json.dumps({k: report[k] for k in ('status', 'production_ready', 'episodes', 'frames', 'blockers')}, indent=2))
        return 0 if stats['production_ready'] else 2
    except (ValueError, KeyError, OSError) as exc:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'normalization_failure.json', {'status': 'BLOCKED', 'reason': str(exc)})
        print(f'BLOCKED: {exc}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
