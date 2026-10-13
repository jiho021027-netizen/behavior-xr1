#!/usr/bin/env python3
"""Data-host-only export of selected train episodes from local LeRobot parquet.

Requires pyarrow. Does not access videos/network or infer the train split.
The explicitly supplied train episode inventory must come from dataset metadata
and the team's actual train/held-out split, not from this script's guesses.
"""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from compute_normalization import TASK_IDS, require, write_json

def export(dataset_root, revision, train_ids_path, task_map_path, output_dir):
    import pyarrow.parquet as pq
    root, out = Path(dataset_root).resolve(), Path(output_dir).resolve()
    ids = json.loads(Path(train_ids_path).read_text())
    require(isinstance(ids,list) and ids and all(type(i) is int and i>=0 for i in ids),
            'train episode inventory must be a nonempty JSON array of integer IDs')
    require(len(set(ids)) == len(ids), 'duplicate train episode inventory ID')
    mapping = json.loads(Path(task_map_path).read_text())
    require(mapping.get('status') == 'VERIFIED_BY_TEAM', 'task mapping must be verified from meta/tasks')
    forward = mapping['official_task_id_to_dataset_task_index']
    require(set(forward) == {str(t) for t in TASK_IDS}, 'task mapping must cover exactly selected 10 tasks')
    require(len(set(forward.values()))==10, 'dataset task indices must be distinct')
    reverse = {int(v):int(k) for k,v in forward.items()}
    require(len(revision)==40 and all(c in '0123456789abcdef' for c in revision), 'revision must be a pinned SHA')
    info = json.loads((root/'meta/info.json').read_text())
    for key,shape in [('observation.state',[61]),('action',[23])]:
        feature=info['features'][key]
        require(feature['shape']==shape and feature['dtype']=='float32', f'{key} metadata mismatch')
    shards = sorted((root/'data').rglob('*.parquet'))
    require(bool(shards),'local data parquet not found')
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/'raw_manifest.json').exists(), 'use a fresh output directory')
    needed = set(ids); parts = {i:[] for i in ids}; source_shards={i:set() for i in ids}
    cols=['observation.state','action','episode_index','task_index','frame_index','timestamp']
    # Spool filtered numeric fragments to disk so memory never holds all 10-task data.
    with tempfile.TemporaryDirectory(prefix='xr1-raw-spool-',dir=out) as tmp:
        counter=0
        for shard in shards:
            pf=pq.ParquetFile(shard)
            require(set(cols)<=set(pf.schema_arrow.names), f'{shard}: required columns missing')
            for batch in pf.iter_batches(batch_size=4096,columns=cols):
                b=batch.to_pydict()
                eps=np.asarray(b['episode_index'],dtype=np.int64)
                tasks=np.asarray(b['task_index'],dtype=np.int64)
                selected=np.isin(eps,ids)
                if not selected.any():continue
                require(all(int(t) in reverse for t in tasks[selected]),
                        'train episode inventory includes a task outside selected 10')
                for eid in np.unique(eps[selected]):
                    indices=np.flatnonzero(eps==eid)
                    tasks_here=np.unique(tasks[indices]);require(len(tasks_here)==1,'episode mixes task IDs')
                    f=Path(tmp)/f'{counter}.npz';counter+=1
                    np.savez(f, **{
                        'observation.state':np.asarray([b['observation.state'][j] for j in indices],dtype=np.float32),
                        'action':np.asarray([b['action'][j] for j in indices],dtype=np.float32),
                        'frame_index':np.asarray([b['frame_index'][j] for j in indices],dtype=np.int64),
                        'timestamp':np.asarray([b['timestamp'][j] for j in indices],dtype=np.float64),
                        'task_id':np.asarray(reverse[int(tasks_here[0])])})
                    parts[int(eid)].append(f)
                    source_shards[int(eid)].add(str(shard.relative_to(root)))
        records=[];missing=[];damaged=[]
        for eid in ids:
            if not parts[eid]:missing.append(eid);continue
            try:
                values={k:[] for k in ('observation.state','action','frame_index','timestamp')};seen_tasks=set()
                for f in parts[eid]:
                    with np.load(f,allow_pickle=False) as piece:
                        seen_tasks.add(int(piece['task_id']))
                        for k in values:values[k].append(piece[k])
                require(len(seen_tasks)==1,'episode has multiple official task IDs')
                values={k:np.concatenate(v) for k,v in values.items()}
                order=np.argsort(values['frame_index']);values={k:v[order] for k,v in values.items()}
                n=len(order)
                require(np.array_equal(values['frame_index'],np.arange(n)), 'duplicate/missing episode frame')
                require(values['observation.state'].shape==(n,61) and values['action'].shape==(n,23), 'raw shape mismatch')
                require(all(np.isfinite(v).all() for v in values.values()), 'nonfinite raw rows')
                filename=f'raw_episode_{eid:06d}.npz';np.savez(out/filename,**values)
                records.append({'episode_id':eid,'task_id':seen_tasks.pop(),'split':'train',
                    'frames':n,'raw_npz':filename,'source_shards':sorted(source_shards[eid])})
            except (ValueError,KeyError,OSError) as exc:
                damaged.append({'episode_id':eid,'reason':str(exc)})
    manifest={'dataset_repo':'behavior-1k/2026-challenge-demos','dataset_revision':revision,
        'official_task_ids':TASK_IDS,'source_data_kind':'real_2026_demo','episodes':records,
        'expected_train_episode_ids':ids,'missing_episodes':missing,'damaged_episodes':damaged,
        'split_inventory_source':str(Path(train_ids_path).resolve()),
        'task_map_source':str(Path(task_map_path).resolve()),'video_accessed':False}
    write_json(out/'raw_manifest.json',manifest)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True);p.add_argument('--dataset-revision',required=True)
    p.add_argument('--train-episode-ids',required=True);p.add_argument('--task-map',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    try:
        m=export(a.dataset_root,a.dataset_revision,a.train_episode_ids,a.task_map,a.output_dir)
        print(json.dumps({'episodes':len(m['episodes']),'missing':m['missing_episodes'],'damaged':m['damaged_episodes']},indent=2))
    except (ValueError,KeyError,OSError,ImportError) as exc:
        raise SystemExit(f'BLOCKED_DATA_EXPORT: {exc}')
