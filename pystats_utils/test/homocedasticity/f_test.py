from scipy.stats import f as fDistribution
import numpy as np

from pystats_utils.test.homocedasticity import Homocedasticity


class FTest(Homocedasticity):

    def runTest(self, workingData: list) -> dict:

        results = {}

        for numeratorSubset in workingData:
            for denominatorSubset in workingData:
                if numeratorSubset != denominatorSubset:

                    numeratorVar = np.var(np.array(numeratorSubset), ddof = 1)
                    denominatorVar = np.var(np.array(denominatorSubset), ddof = 1)

                    if numeratorVar < denominatorVar:
                        numeratorSubset, denominatorSubset = denominatorSubset, numeratorSubset
                        numeratorVar, denominatorVar = denominatorVar, denominatorVar

                    statistic = numeratorVar / denominatorVar

                    pvalue = 2 * (1 - fDistribution.cdf(statistic,
                                                        len(numeratorSubset) - 1,
                                                        len(denominatorSubset) - 1))

                    try:
                        if pvalue < results["pvalue"]:
                            results["statistic"], results["pvalue"] = statistic, pvalue

                    except KeyError:
                        results["statistic"], results["pvalue"] = statistic, pvalue

        return results