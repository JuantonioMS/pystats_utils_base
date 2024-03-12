from scipy.stats import normaltest

from pystats_utils.statistical_test.univariate.normality.normality import NormalityTest

class AgostinoTest(NormalityTest):



    @staticmethod
    def _coreFunction(data):
        return normaltest(data)