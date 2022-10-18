from scipy.stats import ttest_ind

from pystats_utils.test.value_comparison import ValueComparison


class WelchTest(ValueComparison):


    def runTest(self, workingData: list) -> dict:

        results = {}

        for subset1 in workingData:
            for subset2 in workingData:
                if subset1 != subset2:

                    statistic, pvalue = ttest_ind(subset1,
                                                  subset2,
                                                  equal_var = False)

                    try:
                        if pvalue < results["pvalue"]:
                            results["statistic"], results["pvalue"] = statistic, pvalue

                    except KeyError:
                        results["statistic"], results["pvalue"] = statistic, pvalue

        return results