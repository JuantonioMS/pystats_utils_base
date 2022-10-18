from scipy.stats import wilcoxon

from pystats_utils.test.value_comparison import ValueComparison


class WilocoxonSignedRankTest(ValueComparison):


    def runTest(self, workingData: list) -> dict:

        results = {}

        for subset1 in workingData:
            for subset2 in workingData:
                if subset1 != subset2:

                    statistic, pvalue = wilcoxon(subset1, subset2)

                    try:
                        if pvalue < results["pvalue"]:
                            results["statistic"], results["pvalue"] = statistic, pvalue

                    except KeyError:
                        results["statistic"], results["pvalue"] = statistic, pvalue

        return results