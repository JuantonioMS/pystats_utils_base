from scipy.stats import bartlett

from pystats_utils.statistical_test.bivariate.homocedasticity.homocedasticity import Homocedasticity

class BartlettTest(Homocedasticity):



    @staticmethod
    def _coreFunction(values1, values2):

        statistic, pvalue = bartlett(values1,
                                     values2)

        return statistic, pvalue