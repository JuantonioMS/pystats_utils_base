from pystats_utils.statistical_test.bivariate.bivariate import BivariateStatisticalTest

from pystats_utils.configuration.key_words import P_VALUE, STATISTIC

class NormalityTest(BivariateStatisticalTest):


    def _runTest(self, data: dict) -> dict:

        results = {}

        for group, values in data.items():
            results[group] = {}

            statistic, pvalue = self._coreFunction(values)

            results[group][P_VALUE] = pvalue
            results[group][STATISTIC] = statistic

        return results