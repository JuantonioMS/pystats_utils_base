from scipy.stats import kstest
from scipy.stats import norm

from pystats_utils.statistical_test.univariate.normality.normality import NormalityTest

class KolmogorovSmirnovTest(NormalityTest):



    @staticmethod
    def _coreFunction(data):
        return kstest(data, norm.cdf)