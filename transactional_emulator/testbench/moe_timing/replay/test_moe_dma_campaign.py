import copy
import json
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unittest

from compare_moe_normal import digest, validate_native
from run_moe_dma_campaign import save, scale_layout


class NativeEvidenceGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.library = Path(self.tmp.name)/'native.so'
        self.library.write_bytes(b'test-library-identity')
        self.arch = dict(clock_period_ps=1000, dma=dict(issue_policy='per_channel', frontend_sram_bytes=45056))
        controllers = [dict(id='Channel '+str(i), num_read_reqs=1, num_read_reqs_served=1,
            num_read_reqs_forwarded=0, num_write_reqs=0) for i in range(2)]
        cal = dict(capi_version=2, native_transaction_bytes=32, mapper='CacheLineInterleave',
            channel_shift=5, channels=2, issue_policy='per_channel', issue_period_ps=1000,
            native_pending=0, native_inflight_peak=2, submission_entries=256,
            accepted_per_channel=[1,1], rejected_per_channel=[0,0],
            native_stats=dict(memory_system=dict(controller=controllers, total_num_read_requests=2)))
        dma = dict(reserved_bytes=16384, line_requests=2, sector_requests=3, merged_sectors=1,
            useful_copy_bytes=64, lookup_busy_ps=4000, copy_busy_ps=2000, mshr_peak=1)
        self.envelope = dict(memory_model=dict(calibration=cal),
            result=dict(hbm_read_bytes=64, global_dma_inflight_peak=2, total_ps=100000, dma_frontend=dma),
            provenance=dict(native_library_path=str(self.library), native_library_sha256=digest(self.library)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_reconciled_native_counts_pass(self):
        validate_native(self.envelope, self.arch, 2)

    def test_incorrect_native_or_dma_evidence_fails_closed(self):
        mutations = [
            (['memory_model','calibration','native_transaction_bytes'],16),
            (['memory_model','calibration','native_pending'],1),
            (['memory_model','calibration','native_stats','memory_system','controller',0,'num_read_reqs_served'],0),
            (['memory_model','calibration','accepted_per_channel'],[1,2]),
            (['result','hbm_read_bytes'],128),
            (['result','dma_frontend','merged_sectors'],2),
            (['result','dma_frontend','reserved_bytes'],45057),
            (['result','dma_frontend','mshr_peak'],3),
            (['result','dma_frontend','copy_busy_ps'],0),
            (['provenance','native_library_sha256'],'wrong'),
        ]
        for path, value in mutations:
            with self.subTest(path=path):
                e=copy.deepcopy(self.envelope); node=e
                for key in path[:-1]: node=node[key]
                node[path[-1]]=value
                with self.assertRaises(ValueError): validate_native(e,self.arch,2)


class LayoutPermutation(unittest.TestCase):
    def test_scale_padding_preserves_every_element_and_scale_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; target=Path(tmp)/'target'
            source.mkdir()
            image=bytes((i*7)%256 for i in range(384))
            (source/'weights.bin').write_bytes(image)
            expert=dict(id=0)
            for index,name in enumerate(['gate','up','down']):
                expert[name]=dict(rows=2,cols=8,element_base=128*index,scale_base=128*index+64,
                                  element_row_stride=8,scale_row_stride=1)
            manifest=dict(hbm_file='weights.bin',experts=[expert],metadata=dict(hbm_sha256=digest(source/'weights.bin')))
            save(source/'workload.json',manifest)
            golden=dict(output_f32=[[1.0]],output_bf16=[[16256]],workload_sha256=digest(source/'workload.json'),hbm_sha256=digest(source/'weights.bin'))
            save(source/'golden.json',golden)
            scale_layout(source,target)
            new=json.loads((target/'workload.json').read_text())
            data=(target/'weights.bin').read_bytes()
            for name in ['gate','up','down']:
                old=expert[name]; region=new['experts'][0][name]
                self.assertEqual(region['scale_row_stride'],32)
                for stream,width in [('element',8),('scale',1)]:
                    for row in range(2):
                        a=old[stream+'_base']+row*old[stream+'_row_stride']
                        b=region[stream+'_base']+row*region[stream+'_row_stride']
                        self.assertEqual(image[a:a+width],data[b:b+width])
            self.assertEqual(json.loads((target/'golden.json').read_text())['output_f32'],golden['output_f32'])
            self.assertEqual(new['metadata']['hbm_sha256'],digest(target/'weights.bin'))
            self.assertEqual(digest(source/'weights.bin'),manifest['metadata']['hbm_sha256'])


if __name__ == '__main__': unittest.main()
