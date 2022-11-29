from scipy.stats import bartlett

from pystats_utils.test.homocedasticity import Homocedasticity

class BartlettTest(Homocedasticity):

    def runTest(self, data: dict) -> dict:

        results = {"pvalue"    : {},
                   "statistic" : {}}

        for group1 in data:
            for group2 in data:

                if group1 != group2:

                    title = " vs. ".join(sorted([group1, group2]))

                    if title in results["pvalue"]:
                        continue

                    statistic, pvalue = bartlett(data[group1],
                                                 data[group2])

                    results["pvalue"][title] = pvalue
                    results["statistic"][title] = statistic

        return results