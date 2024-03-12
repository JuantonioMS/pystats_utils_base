from scipy.stats import mannwhitneyu

from pystats_utils.statistical_test.bivariate.value_comparison.value_comparison import ValueComparisonTest


class MannWhitneyUTest(ValueComparisonTest):



    @staticmethod
    def _coreFunction(values1, values2):

        statistic, pvalue = mannwhitneyu(values1,
                                         values2)

        return statistic, pvalue