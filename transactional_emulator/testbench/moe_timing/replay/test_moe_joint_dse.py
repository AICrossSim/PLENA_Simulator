import copy
import unittest
from run_moe_joint_dse import (shapes, category, architecture, resource_gate,
                               admission, equivalence_key, frontend_bytes)

class JointDseTests(unittest.TestCase):
    def test_complete_topologies_and_resource_totals(self):
        grid=shapes()
        self.assertEqual(len(grid),37)
        self.assertEqual(len(set(grid)),37)
        counts={c:sum(category(s)==c for s in grid) for c in
                ['single','homogeneous','equal_pe_different_shape','large_small']}
        self.assertEqual(counts,dict(single=6,homogeneous=5,equal_pe_different_shape=10,large_small=16))
        for shape in grid:
            for allocation in ['equal','pe','inverse_pe']:
                resource_gate(architecture(shape,2,64,8,allocation))

    def test_sram_boundary_includes_dma_tile_descriptors(self):
        a=architecture(((64,64),),4,128,8,'equal')
        self.assertGreater(frontend_bytes(a),45056)
        with self.assertRaisesRegex(ValueError,'descriptor'):resource_gate(a)
        a['cores'][0]['weight_slots']=3
        resource_gate(a)

    def test_equivalence_is_only_when_actual_admission_is_identical(self):
        shape=((16,192),(8,128))
        a=architecture(shape,4,128,8,'equal')
        b=architecture(shape,4,128,8,'pe')
        def w(m):return dict(input_dim=2048,expert_hidden_dim=1408,
                            routes=[dict(expert=0) for _ in range(m)],inputs_bf16=[[]]*m)
        ma={'fixture':admission(a,w(30))}; mb={'fixture':admission(b,w(30))}
        self.assertEqual(equivalence_key(a,ma,'raw'),equivalence_key(b,mb,'raw'))
        ma={'fixture':admission(a,w(32))}; mb={'fixture':admission(b,w(32))}
        self.assertNotEqual(equivalence_key(a,ma,'raw'),equivalence_key(b,mb,'raw'))
        self.assertNotEqual(equivalence_key(a,ma,'raw'),equivalence_key(a,ma,'scale_odd32'))
        changed=copy.deepcopy(a);changed['dispatch_threshold']=1
        self.assertNotEqual(equivalence_key(a,ma,'raw'),equivalence_key(changed,ma,'raw'))


class FinalistEvidenceTests(unittest.TestCase):
    def test_repeat_signature_retains_every_value_time_and_native_counter(self):
        from validate_moe_dse_finalists import numerical_result_signature
        a=dict(result=dict(architecture='a',total_ps=100,output_bf16=[[42]],hbm_read_bytes=64),
               memory_model=dict(native_pending=0,accepted=2))
        b=copy.deepcopy(a);b['result']['architecture']='b'
        self.assertEqual(numerical_result_signature(a),numerical_result_signature(b))
        b['result']['total_ps']=101
        self.assertNotEqual(numerical_result_signature(a),numerical_result_signature(b))
        b=copy.deepcopy(a);b['memory_model']['accepted']=3
        self.assertNotEqual(numerical_result_signature(a),numerical_result_signature(b))
        b=copy.deepcopy(a);b['result']['output_bf16']=[[43]]
        self.assertNotEqual(numerical_result_signature(a),numerical_result_signature(b))

    def test_incomplete_grid_cannot_publish_a_conclusion(self):
        import json,tempfile
        from pathlib import Path
        from report_moe_joint_dse import report,geomean
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'campaign.json').write_text(json.dumps(dict(expected_runs=4)))
            (root/'status.json').write_text(json.dumps(dict(status='running',passed=3,expected=4)))
            with self.assertRaisesRegex(ValueError,'incomplete'):report(root)
            self.assertFalse((root/'RESULT_ZH.md').exists())
        self.assertAlmostEqual(geomean([1,4]),2)
        for values in [[],[0],[float('nan')],[float('inf')],[-1]]:
            with self.assertRaises(ValueError):geomean(values)

if __name__=='__main__':unittest.main()
