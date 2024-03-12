from scipy.stats import ttest_ind

from pystats_utils.statistical_test.bivariate.value_comparison.value_comparison import ValueComparisonTest

class WelchTest(ValueComparisonTest):



    @staticmethod
    def _coreFunction(values1, values2):

        statistic, pvalue = ttest_ind(values1,
                                      values2,
                                      equal_var = False)

        return statistic, pvalue