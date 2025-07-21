import unittest
import pandas as pd
from bp_bayesian_utils import bp_dict, get_result_df
from tnreason.engine import workload_to_pandas as wtp
from tnreason import engine


class ToyAccountingBPTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(bp_dict)

        # Define labels and features
        self.labels = ['ACCOUNT_tV', 'BP_ACCOUNT_tV']
        self.features = ['BUSINESSPARTNER_tV', 'VAT_tV']

        # Create TermCore object
        self.termCore = wtp.PandasTermCore(valueDf=self.df)
        self.pdCore = engine.convert(self.termCore, "PolynomialCore")

    def test_script_execution(self):
        """Test if the script runs successfully."""
        # Core contraction
        smallCore = engine.contract(
            coreDict={"pdCore": self.pdCore.clone()},
            openColors=self.features + self.labels,
            contractionMethod="CorewiseContractor"
        )
        smallCore.add_identical_slices()

        normedCore = engine.normalize(
            {"core": smallCore.clone()},
            outColors=self.labels,
            inColors=self.features,
            contractionMethod="CorewiseContractor",
            coreType="PolynomialCore"
        )

        resultDf = get_result_df(normedCore.values, smallCore.values, self.termCore)

        assert not resultDf.empty, "The result DataFrame is empty."
        assert resultDf.shape[0] == 5, "The DataFrame does not have 5 rows."
        assert resultDf[(resultDf["BUSINESSPARTNER"]=="24") & (resultDf["VAT"]=="0")]["probability %"].iloc[0] == 94, "The probability for tested row is not 94"
        assert "3" not in resultDf["VAT"].values, "Value 3 is unexpectedly found in column 'VAT'."



if __name__ == '__main__':
    unittest.main()