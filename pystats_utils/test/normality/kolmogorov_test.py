from scipy.stats import kstest
from scipy.stats import norm

from pystats_utils.test.normality import Normality

class KolmogorovSmirnovTest(Normality):

    def runTest(self, workingData: list) -> dict:

        results = {}

        for subset in workingData:

            result = kstest(subset, norm.cdf)
            statistic, pvalue = result.statistic, result.pvalue

            try:
                if pvalue < results["pvalue"]:
                    results["statistic"], results["pvalue"] = statistic, pvalue

            except KeyError:
                results["statistic"], results["pvalue"] = statistic, pvalue

        return results