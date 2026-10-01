"""Scope and input-pairing checks for the BF16 A/B study."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ipd_ab_study
import multilayer_profile


class IpdAbStudyTests(unittest.TestCase):
    def test_full_matrix_keeps_identical_resources_and_routes(self):
        points=ipd_ab_study.cases()
        self.assertEqual(len(points),180)
        groups={}
        for p in points:
            key=(p['workload']['id'],p['organization'])
            groups.setdefault(key,[]).append(p)
            self.assertEqual(p['config']['credits'],256)
            self.assertEqual(p['config']['hbm_bytes_per_ns'],256)
            self.assertEqual(p['config']['split'],'none')
            self.assertFalse(p['config']['tail_partition'])
            self.assertEqual(p['resources']['joint_state_bytes'],256)
        for cases in groups.values():
            self.assertEqual({p['mode'] for p in cases},set(ipd_ab_study.POLICIES))
            self.assertEqual({tuple(p['design']['resources']['weight_slots']) for p in cases},
                             {tuple(cases[0]['design']['resources']['weight_slots'])})
            self.assertEqual({tuple(e['id'] for e in p['workload']['experts']) for p in cases},
                             {tuple(e['id'] for e in cases[0]['workload']['experts'])})

    def test_adjacent_layers_use_same_captured_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic_adjacent.npz'
            n=6
            routes=np.zeros((n,1,2,2),dtype=np.int64)
            routes[:,:,0,:]=[0,1]
            routes[:,:,1,:]=[2,3]
            weights=np.full((n,1,2,2),0.5,dtype=np.float32)
            meta='{"top_k":2,"hidden_size":512,"routed_experts":4,"moe_intermediate_size":128,"shared_expert_intermediate_size":128}'
            np.savez(path,meta=np.array(meta),layer_ids=np.array([12,13]),
                     sample_ids=np.array([f'ipd-ab-synthetic-{i}' for i in range(n)]),
                     valid=np.ones((n,1),dtype=bool),decode_idx=routes,decode_weight=weights)
            layers,source=multilayer_profile.captured_layers(path,[12,13],0,2)
            self.assertEqual(len(layers),2)
            self.assertEqual(len(source['sample_ids']),2)
            self.assertEqual([t['sample_id'] for t in layers[0]['tokens']],
                             [t['sample_id'] for t in layers[1]['tokens']])
            self.assertEqual({e['id'] for e in layers[0]['experts'] if not e['is_shared']},{0,1})
            self.assertEqual({e['id'] for e in layers[1]['experts'] if not e['is_shared']},{2,3})
            first={weight['hbm_base'] for e in layers[0]['experts'] for weight in e['weights'].values()}
            second={weight['hbm_base'] for e in layers[1]['experts'] for weight in e['weights'].values()}
            self.assertFalse(first & second)


if __name__=='__main__':unittest.main()
