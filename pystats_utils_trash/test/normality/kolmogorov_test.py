from scipy.stats import kstest
from scipy.stats import norm

from pystats_utils.test.normality import Normality

class KolmogorovSmirnovTest(Normality):


    def runTest(self, data) -> dict:

        results = {}

        if isinstance(data, list):

            results = {}

            aux = kstest(data, norm.cdf)

            results["statistic"], results["pvalue"] = aux.statistic, aux.pvalue

        else:

            results = {"pvalue"    : {},
                       "statistic" : {}}

            for group in data:

                results[group] = {}

                aux = kstest(data[group], norm.cdf)

                results["statistic"][group], results["pvalue"][group] = aux.statistic, aux.pvalue

        return results