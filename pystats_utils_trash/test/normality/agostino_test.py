from scipy.stats import normaltest

from pystats_utils.test.normality import Normality

class AgostinoTest(Normality):


    def runTest(self, data) -> dict:

        if isinstance(data, list):

            results = {}

            aux = normaltest(data)

            results["statistic"], results["pvalue"] = aux.statistic, aux.pvalue

        else:

            results = {"pvalue"    : {},
                       "statistic" : {}}

            for group in data:


                aux = normaltest(data[group])


                results["statistic"][group], results["pvalue"][group] = aux.statistic, aux.pvalue

        return results