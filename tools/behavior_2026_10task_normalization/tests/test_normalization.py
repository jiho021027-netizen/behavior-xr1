"""Small CPU synthetic checks; these are NOT 2026 dataset validation."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from compute_normalization import TASK_IDS, STATE_ACTIVE, ACTION_ACTIVE, compute, write_json
from reference.r1pro_2026_state import adapt_r1pro_2026_state

class Tests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.default_rng(1012)
        episodes, states, actions, masks = [], [], [], []
        for i,t in enumerate(TASK_IDS):
            n = 33+i
            s = np.zeros((n,60),np.float32)
            s[:,STATE_ACTIVE] = rng.normal(size=(n,20))
            s[:,0] = 1  # zero-range state
            a = np.zeros((n,30,60),np.float32)
            a[:,:,ACTION_ACTIVE] = rng.normal(size=(n,30,18))
            a[:,:,6] = 2  # constant gripper channel
            m = np.zeros_like(a, dtype=bool)
            valid = np.arange(n)[:,None]+np.arange(1,31)[None,:] < n
            m[:,:,ACTION_ACTIVE] = valid[:,:,None]
            a[~m] = np.nan  # invalid values must not leak into stats
            ep = {'task_id':t, 'episode_id':i, 'frames':n, 'split':'train'}
            for k,v in [('states',s),('actions',a),('mask',m),
                         ('frame_index',np.arange(n)),('timestamp',np.arange(n)/30.0)]:
                ep[k] = f'{i}_{k}.npy'
                np.save(root/ep[k], v)
            episodes.append(ep); states.append(s); actions.append(a); masks.append(m)
        for name in ('state_schema','action_schema','target_builder'):
            (root/name).write_text('SYNTHETIC TEST ONLY')
        manifest = {'dataset_repo':'behavior-1k/2026-challenge-demos', 'dataset_revision':'a'*40,
            'official_task_ids':TASK_IDS, 'episodes':episodes,
            'state_schema':'state_schema','action_schema':'action_schema','target_builder':'target_builder',
            'task_id_to_embedding_index':{str(t):t for t in TASK_IDS},
            'target_offset':'h+1_from_current_observation', 'source_data_kind':'synthetic',
            'expected_train_episode_ids':list(range(10)), 'missing_episodes':[], 'damaged_episodes':[],
            'sampling_population':'one_state_and_one_anchor_per_frame'}
        write_json(root/'train_manifest.json',manifest)
        return manifest,np.concatenate(states),np.concatenate(actions),np.concatenate(masks)

    def test_streaming_against_independent_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,s,a,m=self.fixture(root)
            stats,report=compute(root/'train_manifest.json',root/'out',block_size=7)
            mean,std=np.asarray(stats['action_mean']),np.asarray(stats['action_std'])
            for h in range(30):
                for c in ACTION_ACTIVE:
                    values=a[:,h,c][m[:,h,c]].astype(np.float64)
                    np.testing.assert_allclose(mean[h,c],values.mean(),atol=1e-12)
                    np.testing.assert_allclose(std[h,c],values.std(),atol=1e-12)
            np.testing.assert_allclose(np.asarray(stats['state_q01'])[0],np.quantile(s.astype(np.float64),.01,axis=0))
            np.testing.assert_allclose(np.asarray(stats['state_q99'])[0],np.quantile(s.astype(np.float64),.99,axis=0))
            np.testing.assert_array_equal(stats['action_valid_count'],m.sum(axis=0))
            self.assertFalse(stats['production_ready'])
            self.assertEqual(report['source_data_kind'],'synthetic')
            self.assertTrue((std[:,6]==0).all())
            self.assertIn(0,report['zero_range_state'])

    def test_boundary_leak_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,*_=self.fixture(root)
            path=root/manifest['episodes'][0]['mask']
            m=np.load(path); m[-1,0,ACTION_ACTIVE]=True; np.save(path,m)
            with self.assertRaisesRegex(ValueError,'valid targets|episode tail'):
                compute(root/'train_manifest.json',root/'out')

    def test_duplicate_and_unselected_task_rejected(self):
        for mutation in ('duplicate','wrong_task'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); manifest,*_=self.fixture(root)
                if mutation=='duplicate':manifest['episodes'][1]['episode_id']=0
                else:manifest['episodes'][1]['task_id']=99
                write_json(root/'train_manifest.json',manifest)
                with self.assertRaises(ValueError):compute(root/'train_manifest.json',root/'out')

    def test_mask_reserved_channel_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,*_=self.fixture(root)
            path=root/manifest['episodes'][0]['mask']
            m=np.load(path);m[0,0,59]=True;np.save(path,m)
            with self.assertRaises(ValueError):compute(root/'train_manifest.json',root/'out')

    def test_state_adapter_exact_mapping(self):
        raw=np.arange(61,dtype=np.float32)
        actual=adapt_r1pro_2026_state(raw)
        np.testing.assert_array_equal(actual[:7],raw[3:10])
        np.testing.assert_array_equal(actual[8:15],raw[28:35])
        np.testing.assert_array_equal(actual[20:24],raw[53:57])
        self.assertEqual(actual[7],24.5);self.assertEqual(actual[15],49.5)
        self.assertTrue((actual[[i for i in range(60) if i not in STATE_ACTIVE]]==0).all())
        with self.assertRaises(ValueError):adapt_r1pro_2026_state(np.zeros(256))

if __name__=='__main__':unittest.main()
