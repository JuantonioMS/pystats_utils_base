from scipy.stats import fligner

from pystats_utils.statistical_test.bivariate.homocedasticity.homocedasticity import HomocedasticityTest

class FlignerTest(HomocedasticityTest):



    @staticmethod
    def _coreFunction(values1, values2):

        statistic, pvalue = fligner(values1,
                                   values2,
                                   center = "median")

        return statistic, pvalue