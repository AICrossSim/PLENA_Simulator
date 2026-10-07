#!/usr/bin/env python3
"""Repeat the actual full-prompt CPU model prefix into an isolated directory."""
import tempfile,shutil,json
from pathlib import Path
import torch
import capture_real as c

def main():
    original=c.OUT
    with tempfile.TemporaryDirectory(prefix='plena-prefix-repeat-',dir='/tmp') as tmp:
        c.OUT=Path(tmp);(c.OUT/'inputs').mkdir();c.CACHE=c.OUT/'weights';c.CACHE.mkdir()
        shutil.copy2(original/'inputs/BFCL_v3_simple.json',c.OUT/'inputs/BFCL_v3_simple.json')
        c.main()
        a=torch.load(original/'real_inputs/capture.pt',weights_only=True,map_location='cpu')
        b=torch.load(c.OUT/'real_inputs/capture.pt',weights_only=True,map_location='cpu')
        for key in ['x','routes','route_weights']:assert torch.equal(a[key],b[key]),key
        m=json.loads((original/'real_inputs/manifest.json').read_text())
        n=json.loads((c.OUT/'real_inputs/manifest.json').read_text())
        assert m['examples']==n['examples'] and m['tensor_provenance']==n['tensor_provenance']
        assert m['x']['sha256']==n['x']['sha256']
        for key in m['exported_weights']:assert m['exported_weights'][key]['sha256']==n['exported_weights'][key]['sha256']
    (original/'real_inputs/repeat_validation.json').write_text(json.dumps(dict(passed=True,full_prompts=16,
        actual_activation_bit_exact=True,router_ids_and_weights_bit_exact=True,original_tensor_hashes_identical=True,
        captured_phase='last-token prefill, layer1 MoE input'),indent=2)+'\n')
    print('PREFIX REPEAT PASS')
if __name__=='__main__':main()
