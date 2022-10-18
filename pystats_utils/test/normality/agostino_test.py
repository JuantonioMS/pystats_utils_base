from scipy.stats import normaltest

from pystats_utils.test.normality import Normality

class AgostinoTest(Normality):

    def runTest(self, workingData: list) -> dict:

        results = {}

        for subset in workingData:

            statistic, pvalue = normaltest(subset)

            try:
                if pvalue < results["pvalue"]:
                    results["statistic"], results["pvalue"] = statistic, pvalue

            except KeyError:
                results["statistic"], results["pvalue"] = statistic, pvalue

        return results