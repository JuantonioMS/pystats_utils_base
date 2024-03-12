from pystats_utils.statistical_test.univariate.univariate import UnivariateStatisticalTest

from pystats_utils.configuration.key_words import P_VALUE, STATISTIC

class NormalityTest(UnivariateStatisticalTest):



    def _runTest(self, data) -> dict:

        results = {}

        aux = self._coreFunction(data)

        results[STATISTIC], results[P_VALUE] = aux.statistic, aux.pvalue

        return results