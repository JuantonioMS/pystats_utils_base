from scipy.stats import bartlett

from pystats_utils.test.homocedasticity import Homocedasticity

class BartlettTest(Homocedasticity):

    def runTest(self, workingData: list) -> dict:

        results = {}

        results["statistic"], results["pvalue"] = bartlett(*workingData)

        return results