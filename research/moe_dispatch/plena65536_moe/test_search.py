"""Small exhaustive references validate lower-bound geometric pruning."""
import numpy as np

from research.moe_dispatch.plena65536_shared.geometry import Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from research.moe_dispatch.plena65536_moe.model import cost_table, schedule, window_histograms, balanced_score
from research.moe_dispatch.plena65536_moe.search import evaluate_family


def test_pruning_matches_every_exact_pair():
    windows = []
    for batch in (2, 4, 8):
        windows.append({'id':str(batch), 'batch':batch,'experts':[
            {'id':-1,'Me':batch,'H':2048,'F':2816,'is_shared':True},
            *[{'id':i,'Me':m,'H':2048,'F':1408} for i,m in enumerate((1,2,3,5))]]})
    cores = (Core(2,256,1,64),Core(2,128,1,128),Core(2,512,1,32),
             Core(4,256,1,32),Core(8,512,1,8),Core(16,2048),Core(32,1024))
    assert all(c.macs == 32768 for c in cores)
    ids = np.asarray([(a,b) for a in range(len(cores)) for b in range(a,len(cores))],dtype=np.int32)
    types,hist,weights = window_histograms(windows)
    for profile in PROFILES:
        costs = cost_table(cores,types,profile)
        chosen,lb,partial = evaluate_family(cores,ids,costs,hist,weights,profile,chunk=3)
        exact = np.asarray([balanced_score([schedule((cores[a],cores[b]),w,profile)['cycles'] for w in windows],windows)
                            for a,b in ids])
        assert np.all(lb <= exact+1e-6)
        assert abs(chosen['development_GM_cycles']-exact.min()) < 1e-6
        assert np.all(np.isnan(partial) | (np.abs(partial-exact)<1e-6))
        assert np.all(lb[np.isnan(partial)] > exact.min())
