from scipy.stats import levene

from pystats_utils.test.homocedasticity import Homocedasticity

class LeveneTest(Homocedasticity):

    def runTest(self, workingData: list) -> dict:

        results = {}

        results["statistic"], results["pvalue"] = levene(*workingData, center = "mean")

        return results