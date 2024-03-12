from scipy.stats import normaltest

from pystats_utils.statistical_test.bivariate.normality.normality import NormalityTest

class AgostinoTest(NormalityTest):



    @staticmethod
    def _coreFunction(data):

        aux = normaltest(data)

        return aux.statistic, aux.pvalue