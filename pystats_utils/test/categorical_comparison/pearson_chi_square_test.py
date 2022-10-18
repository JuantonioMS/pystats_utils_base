from scipy.stats import chi2_contingency
from scipy.stats.contingency import crosstab

import pandas as pd

from pystats_utils.test.categorical_comparison import CategoricalComparison

class PearsonChiSquareTest(CategoricalComparison):


    def extractData(self,
                    workingDataframe: pd.DataFrame = pd.DataFrame(),
                    classVariable: str = "",
                    targetVariable: str = "") -> pd.DataFrame:

        return workingDataframe


    def runTest(self, workingData: pd.DataFrame) -> dict:

        results = {}

        _, workingData = crosstab(workingData[workingData.columns[0]],
                                  workingData[workingData.columns[1]])

        results["statistic"], results["pvalue"], results["dof"], results["expected"] = chi2_contingency(workingData)

        return results