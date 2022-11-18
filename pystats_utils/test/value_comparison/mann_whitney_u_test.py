from scipy.stats import mannwhitneyu

from pystats_utils.test.value_comparison import ValueComparison


class MannWhitneyUTest(ValueComparison):


    def runTest(self, workingData: list) -> dict:

        results = {}

        for subset1 in workingData:
            print("................")
            for subset2 in workingData:
                if subset1 != subset2:

                    statistic, pvalue = mannwhitneyu(subset1, subset2)

                    print(pvalue)

                    try:
                        if pvalue < results["pvalue"]:
                            results["statistic"], results["pvalue"] = statistic, pvalue

                    except KeyError:
                        results["statistic"], results["pvalue"] = statistic, pvalue

        return results