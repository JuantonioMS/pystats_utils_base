import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.utils import resample

from pystats_utils.test.categorical_comparison import CategoricalComparison
from pystats_utils.data_operations import isCategorical


class ConfussionMatrixAnalysis(CategoricalComparison):



    def run(self, bootstrapping = 2000):

        self.bootstrapping = bootstrapping

        return super().run()



    def cookData(self, dataframe):

        for variable in [self.classVariable, self.targetVariable]:

            if isCategorical(dataframe, self.classVariable):
                dataframe[variable] = pd.get_dummies(dataframe[variable],
                                                     drop_first = True)

        return dataframe



    def runTest(self, data):

        results = {}

        results["accuracy"] = accuracy_score(data[self.classVariable],
                                             data[self.targetVariable])

        results["error"] = 1 -  results["accuracy"]

        results["precision"] = precision_score(data[self.classVariable],
                                               data[self.targetVariable])

        results["recall"] = recall_score(data[self.classVariable],
                                         data[self.targetVariable])

        results["f1"] = f1_score(data[self.classVariable],
                                 data[self.targetVariable])


        accuracies, errors, precisions, recalls, f1s = [], [], [], [], []
        for _ in range(self.bootstrapping):

            bootResult = ConfussionMatrixAnalysis(dataframe = resample(data),
                                                  classVariable = self.classVariable,
                                                  targetVariable = self.targetVariable).run(bootstrapping = 0)

            accuracies.append(bootResult.accuracy)
            errors.append(bootResult.error)
            precisions.append(bootResult.precision)
            recalls.append(bootResult.recall)
            f1s.append(bootResult.f1)

        if self.bootstrapping:

            values = [results["accuracy"], results["error"], results["precision"], results["recall"], results["f1"]]
            boots = [accuracies, errors, precisions, recalls, f1s]

            lowerCI = [original - 1.97 * np.std(boot)   for original, boot in zip(values, boots)]
            lowerCI = [ci if ci > 0 else 0.0 for ci in lowerCI]

            upperCI = [original + 1.97 * np.std(boot)   for original, boot in zip(values, boots)]
            upperCI = [ci if ci < 1 else 1.0 for ci in upperCI]

            summary = pd.DataFrame({"Value"    : values,
                                    "CI 2.5%"  : lowerCI,
                                    "CI 97.5%" : upperCI},
                                    index = ["Accuracy",
                                             "Error",
                                             "Precision",
                                             "Recall",
                                             "F1"])

            results["summary"] = summary


        return results