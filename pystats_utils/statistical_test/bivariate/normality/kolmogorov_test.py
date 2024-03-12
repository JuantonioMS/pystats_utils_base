from scipy.stats import kstest
from scipy.stats import norm

from pystats_utils.statistical_test.bivariate.normality.normality import NormalityTest

class KolmogorovSmirnovTest(NormalityTest):



    @staticmethod
    def _coreFunction(data):

        aux = kstest(data, norm.cdf)

        return aux.statistic, aux.pvalue