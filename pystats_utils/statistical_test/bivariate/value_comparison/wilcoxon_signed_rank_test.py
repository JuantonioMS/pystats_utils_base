from scipy.stats import wilcoxon

from pystats_utils.statistical_test.bivariate.value_comparison.value_comparison import ValueComparisonTest

class WilcoxonSignedRankTest(ValueComparisonTest):



    @staticmethod
    def _coreFunction(values1, values2):

        statistic, pvalue = wilcoxon(values1,
                                     values2)

        return statistic, pvalue