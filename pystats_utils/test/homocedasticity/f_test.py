from scipy.stats import f as fDistribution
import numpy as np

from pystats_utils.test.homocedasticity import Homocedasticity


class FTest(Homocedasticity):


    def runTest(self, data: dict) -> dict:

        results = {"pvalue"    : {},
                   "statistic" : {}}

        for group1 in data:
            for group2 in data:

                if group1 != group2:

                    title = " vs. ".join(sorted([group1, group2]))

                    if title in results["pvalue"]:
                        continue

                    group1Var = np.var(data[group1])
                    group2Var = np.var(data[group2])

                    if group1Var > group2Var:

                        statistic = group1Var / group2Var

                        pvalue = 2 * (1 - fDistribution.cdf(statistic,
                                                            len(data[group1]) - 1,
                                                            len(data[group2]) - 1))

                    else:

                        statistic = group2Var / group1Var

                        pvalue = 2 * (1 - fDistribution.cdf(statistic,
                                                            len(data[group2]) - 1,
                                                            len(data[group1]) - 1))

                    results["pvalue"][title] = pvalue
                    results["statistic"][title] = statistic

        return results


