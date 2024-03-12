from scipy.stats import chi2_contingency
from scipy.stats.contingency import crosstab

import pandas as pd

from pystats_utils.test.categorical_comparison import CategoricalComparison

class PearsonChiSquareTest(CategoricalComparison):


    def cookData(self, dataframe):

        return dataframe


    def runTest(self, data: pd.DataFrame) -> dict:

        results = {}

        _, contigenceTab = crosstab(list(data[self.classVariable]),
                                    list(data[self.targetVariable]))

        results["statistic"], results["pvalue"], results["dof"], results["expected"] = chi2_contingency(contigenceTab)

        return results