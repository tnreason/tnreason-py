import unittest

from tnreason import engine
from tnreason import encoding

aSuf = encoding.suf.atomicVariableSuffix

class TermLoading(unittest.TestCase):

    def test_pandas_tCore(self):

        import pandas as pd
        from tnreason.engine import workload_to_pandas as wtp

        data = {'id': [1, 2, 3, 4],
                'age': [25, 30, 35, 40],
                'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']}
        df = pd.DataFrame(data)

        tCore = wtp.PandasTermCore()
        tCore.load(df, ["age", "city"])

        self.assertTrue("id" + encoding.suf.termVariableSuffix not in tCore.colors)
        self.assertTrue("age" + encoding.suf.termVariableSuffix in tCore.colors)

        self.assertEqual(engine.convert(tCore, "PolynomialCore")[0, 0], 1)
        self.assertEqual(engine.convert(tCore, "NumpyCore").values.shape[1], 4)
