from scipy.stats import fligner

from pystats_utils.test.homocedasticity import Homocedasticity

class FlignerTest(Homocedasticity):

    def runTest(self, workingData: list) -> dict:

        results = {}

        results["statistic"], results["pvalue"] = fligner(*workingData, center = "median")

        return results