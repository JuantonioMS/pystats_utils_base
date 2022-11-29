from scipy.stats import levene

from pystats_utils.test.homocedasticity import Homocedasticity

class BrownForsythTest(Homocedasticity):


    def runTest(self, data: dict) -> dict:

        results = {"pvalue"    : {},
                   "statistic" : {}}

        for group1 in data:
            for group2 in data:

                if group1 != group2:

                    title = " vs. ".join(sorted([group1, group2]))

                    if title in results["pvalue"]:
                        continue

                    statistic, pvalue = levene(data[group1],
                                               data[group2],
                                               center = "median")

                    results["pvalue"][title] = pvalue
                    results["statistic"][title] = statistic

        return results