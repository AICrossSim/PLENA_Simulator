import unittest
import numpy as np
from .common import population, validate_routes

class Contracts(unittest.TestCase):
    def test_decode_cold_conservation(self):
        ids=np.array([[0,1,2,3,4,5],[6,7,8,9,10,11]])
        validate_routes(ids,np.ones((2,6))/64)
        p=population(ids)
        self.assertEqual(p['cold_experts'],12);self.assertEqual(p['hot_experts'],0)
        self.assertEqual(p['cold_flops_pct'],75);self.assertEqual(p['shared_flops_pct'],25)
        self.assertEqual(p['unique_bf16_weight_MiB'],231)
    def test_hot_uses_assignments_not_expert_count(self):
        ids=np.array([[0,1,2,3,4,5],[0,1,2,3,4,5],[0,6,7,8,9,10]])
        p=population(ids)
        self.assertEqual(p['hot_experts'],1);self.assertEqual(p['hot_assignments'],3)
        self.assertEqual(p['hot_flops_pct'],12.5)
        self.assertEqual(p['hot_flops_pct']+p['cold_flops_pct']+p['shared_flops_pct'],100)
    def test_reject_duplicate_expert(self):
        with self.assertRaises(ValueError):validate_routes(np.array([[0,0,1,2,3,4]]),np.ones((1,6)))
    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError):validate_routes(np.array([[0,1,2,3,4,5]]),np.full((1,6),np.nan))

if __name__=='__main__':unittest.main()
