"""Regressions for the local capacity-gated timer variant, not frozen E4 edits."""
from dataclasses import replace
import sys
import unittest

from ...common import canonical, decode_design, inputs, native_bytes, sha
from ...model import Parameters
from ...predictors import Predictor
from .. import runtime as original
from . import bounded_runtime as bounded


class BoundedTimerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        cls.manifest = json.loads((Path(__file__).resolve().parents[1]/"frozen_designs.json").read_text())
        cls.ws = inputs()

    def design(self, mode, name):
        d = decode_design(self.manifest['modes'][mode][name])
        return replace(d, flows=('WS',)*len(d.cores))

    def test_single_result_remains_byte_exact(self):
        for mode in ('pipelined', 'port_tight'):
            d, p = self.design(mode, 'B1'), Parameters(onchip_mode=mode)
            for batch in (2, 16, 128):
                w = next(w for w in self.ws['heldout'] if w['batch'] == batch)
                a = original.simulate(w, d, p, t_big=3, large_first=True)
                b = bounded.simulate(w, d, p, t_big=3, large_first=True)
                self.assertEqual(canonical(a), canonical(b))

    def test_random_prediction_storm_case_has_finite_pending_state(self):
        # This precise sequential warmup/history hit >7,000 pending events in
        # the unmodified runtime. Observe queue high water without changing it.
        d, p = self.design('pipelined', 'best_hetero'), Parameters()
        pred = Predictor('random')
        target = 'heldout_swe_b16_l2_s11'
        observed = {'peak': 0, 'calls': 0}
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code.co_name == 'event' and 'pending' in frame.f_locals:
                observed['peak'] = max(observed['peak'], len(frame.f_locals['pending'])+1)
                observed['calls'] += 1
        try:
            sys.setprofile(profile)
            for w in self.ws['development']:
                bounded.simulate(w, d, p, predictor=pred, t_big=3, large_first=True)
            for w in self.ws['heldout']:
                result = bounded.simulate(w, d, p, predictor=pred, t_big=3, large_first=True)
                if w['id'] == target:
                    break
            else:
                self.fail('frozen storm regression window missing')
        finally:
            sys.setprofile(None)
        self.assertLess(observed['peak'], 128, observed)
        self.assertGreater(observed['calls'], 0)
        self.assertEqual(len(result['tasks']), len(w['experts']))
        self.assertGreaterEqual(result['hbm_bytes'], native_bytes(w))
        self.assertGreater(result['cycles'], 0)

    def test_original_dispatcher_source_and_callable_unchanged(self):
        self.assertEqual(sha(original.__file__), bounded.ORIGINAL_SOURCE_SHA256)
        self.assertIsNot(original.simulate, bounded.simulate)
        self.assertEqual(original.simulate.__globals__['__name__'], original.__name__)


if __name__ == '__main__':
    unittest.main()
