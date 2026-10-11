"""Regression tests for discovered objective overstatements and exchange failures."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import solver

CASES = json.loads((Path(__file__).parent/'tests/fixtures/missed_minima.json').read_text())['cases']


class OracleRegressionTests(unittest.TestCase):
    def test_all_five_narrow_minima_without_supplied_witness(self):
        for case in CASES:
            with self.subTest(n=case['n']):
                values, sites = solver.oracle(case['points'], grid=39)
                self.assertGreaterEqual(values[0], case['verified_lower']-1e-8)
                self.assertLessEqual(values[0], case['verified_upper']+1e-8)
                self.assertGreater(case['old_false_numerical_objective']-values[0], .05)
                self.assertLess(np.linalg.norm(sites[0]-case['certificate_witness']), 1e-3)

    def test_supplied_witness_is_never_discarded(self):
        for case in CASES:
            p=np.asarray(case['points'])
            # Include interior, boundary, and two nearby witnesses: the old
            # radius-.009 deduplication silently dropped the better of these.
            q=np.asarray(case['certificate_witness'])
            extra=[q+[.0003,-.0002],q,[0.,.123456789]]
            values, sites=solver.oracle(p,grid=11,extra=extra)
            self.assertLessEqual(values[0], np.min(solver.intensity(p,extra))+1e-10)

    def test_rejected_exchange_candidate_preserves_constraints(self):
        sites=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        calls={'oracle':0,'minimize':0}
        def fake_oracle(points,**kwargs):
            calls['oracle']+=1
            return np.full(4,10. if calls['oracle']==1 else 1.),sites.copy()
        def fake_minimize(fun,x0,**kwargs):
            calls['minimize']+=1
            self.assertGreater(len(kwargs['constraints']['fun'](x0)),0)
            x=x0.copy();x[-1]=1./20
            return SimpleNamespace(x=x,nit=1,success=True)
        with patch.object(solver,'oracle',fake_oracle), patch.object(solver,'minimize',fake_minimize), patch.object(solver,'intensity',lambda p,q:np.full(len(q),100.)):
            solver.optimize_exchange(np.array([[.2,.2],[.8,.8]]),rounds=2)
        self.assertEqual(calls['minimize'],2)


if __name__=='__main__':
    unittest.main()
