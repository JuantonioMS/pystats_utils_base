from scipy.stats import levene

from pystats_utils.test.homocedasticity import Homocedasticity

class BrownForsythTest(Homocedasticity):

    def runTest(self, workingData: list) -> dict:

        results = {}

        results["statistic"], results["pvalue"] = levene(*workingData, center = "median")

        return results