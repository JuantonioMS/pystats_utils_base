from scipy.stats import levene

from pystats_utils.statistical_test.bivariate.homocedasticity.homocedasticity import HomocedasticityTest

class LeveneTest(HomocedasticityTest):



    @staticmethod
    def _coreFunction(values1, values2):

        statistic, pvalue = levene(values1,
                                   values2,
                                   center = "mean")

        return statistic, pvalue