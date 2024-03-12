import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.utils import resample

from pystats_utils.test.categorical_comparison import CategoricalComparison


class ConfussionMatrixAnalysis(CategoricalComparison):



    def run(self, bootstrapping = 2000):

        self.bootstrapping = bootstrapping

        return super().run()



    def cookData(self, dataframe):

        for variable in [self.classVariable, self.targetVariable]:

            dataframe[variable] = pd.get_dummies(dataframe[variable],
                                                 drop_first = True if len(set(dataframe[variable])) > 1 else False)

        return dataframe



    def runTest(self, data):

        results = {}

        results["accuracy"] = accuracy_score(data[self.classVariable],
                                             data[self.targetVariable])

        results["error"] = 1 -  results["accuracy"]

        results["precision"] = precision_score(data[self.classVariable],
                                               data[self.targetVariable],
                                             zero_division = 0)

        results["recall"] = recall_score(data[self.classVariable],
                                         data[self.targetVariable],
                                             zero_division = 0)

        results["f1"] = f1_score(data[self.classVariable],
                                 data[self.targetVariable],
                                             zero_division = 0)

        results["tn"], results["fp"], results["fn"], results["tp"] = confusion_matrix(data[self.classVariable],
                                                                                      data[self.targetVariable]).ravel()

        results["specificity"] = results["tn"] / (results["tn"] + results["fp"])
        results["sensitivity"] = results["tp"] / (results["tp"] + results["fn"])

        results["ppv"] = results["tp"] / (results["tp"] + results["fp"])
        results["npv"] = results["tn"] / (results["tn"] + results["fn"])


        accuracies, errors, precisions, recalls, f1s, sensitivities, specificities, ppvs, npvs = [], [], [], [], [], [], [], [], []
        for _ in range(self.bootstrapping):

            bootResult = ConfussionMatrixAnalysis(dataframe = resample(data, stratify = data[self.classVariable]),
                                                  classVariable = self.classVariable,
                                                  targetVariable = self.targetVariable).run(bootstrapping = 0)

            accuracies.append(bootResult.accuracy)
            errors.append(bootResult.error)
            precisions.append(bootResult.precision)
            recalls.append(bootResult.recall)
            f1s.append(bootResult.f1)
            sensitivities.append(bootResult.sensitivity)
            specificities.append(bootResult.specificity)
            ppvs.append(bootResult.ppv)
            npvs.append(bootResult.npv)

        if self.bootstrapping:

            values = [results["accuracy"], results["error"],
                      results["precision"], results["recall"],
                      results["f1"],
                      results["sensitivity"], results["specificity"],
                      results["ppv"], results["npv"]]
            boots = [accuracies, errors,
                     precisions, recalls,
                     f1s,
                     sensitivities, specificities,
                     ppvs, npvs]

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
                                             "F1",
                                             "Sensitivity",
                                             "Specificity",
                                             "PPV",
                                             "NPV"])

            results["summary"] = summary


        return results