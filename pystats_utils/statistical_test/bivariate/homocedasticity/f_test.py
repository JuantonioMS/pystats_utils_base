from scipy.stats import f as fDistribution
import numpy as np

from pystats_utils.statistical_test.bivariate.homocedasticity.homocedasticity import HomocedasticityTest

class FTest(HomocedasticityTest):



    @staticmethod
    def _coreFunction(values1, values2):

        values1Var, values2Var = np.var(values1), np.var(values2)

        if values1Var < values2Var:
            values1Var, values2Var = values2Var, values1Var
            values1, values2 = values2, values1

        statistic = values1Var / values2Var

        pvalue = 2 * (1 - fDistribution.cdf(statistic,
                                            len(values1) - 1,
                                            len(values2) - 1))

        return statistic, pvalue