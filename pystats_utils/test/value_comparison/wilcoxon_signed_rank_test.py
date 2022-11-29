from scipy.stats import wilcoxon

from pystats_utils.test.value_comparison import ValueComparison


class WilocoxonSignedRankTest(ValueComparison):


    def runTest(self, data: dict) -> dict:

        results = {"pvalue"    : {},
                   "statistic" : {}}

        for group1 in data:
            for group2 in data:

                if group1 != group2:

                    title = " vs. ".join(sorted([group1, group2]))

                    if title in results["pvalue"]:
                        continue

                    statistic, pvalue = wilcoxon(data[group1],
                                                 data[group2])

                    results["pvalue"][title] = pvalue
                    results["statistic"][title] = statistic

        return results