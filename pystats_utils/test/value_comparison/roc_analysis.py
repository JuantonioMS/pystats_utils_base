import warnings
warnings.filterwarnings('ignore') 

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.utils import resample

from pystats_utils.test.value_comparison import ValueComparison
from pystats_utils.data_operations import isCategorical


class RocAnalysis(ValueComparison):



    def run(self, bootstrapping = 2000):

        self.bootstrapping = bootstrapping

        return super().run()



    def cookData(self, dataframe):

        if isCategorical(dataframe, self.classVariable):
            dataframe[self.classVariable] = pd.get_dummies(dataframe[self.classVariable],
                                                        drop_first = True)

        return dataframe



    def runTest(self, data):

        results = {}

        aucROC = roc_auc_score(data[self.classVariable],
                               data[self.targetVariable])

        fpr, tpr, thresholds = roc_curve(data[self.classVariable],
                                         data[self.targetVariable])

        info = pd.DataFrame({"FPR"       : fpr,
                             "TPR"       : tpr,
                             "Threshold" : thresholds,
                             "opt"       : tpr + (1 - fpr)})

        cutOff = info['Threshold'].iloc[info['opt'].idxmax()]

        results["aucROC"] = aucROC
        results["cutOff"] = cutOff


        aucROCs, cutOffs = [], []
        for _ in range(self.bootstrapping):

            bootResult = RocAnalysis(dataframe = resample(data),
                                     classVariable = self.classVariable,
                                     targetVariable = self.targetVariable).run(bootstrapping = 0)

            aucROCs.append(bootResult.aucROC)
            cutOffs.append(bootResult.cutOff)


        if self.bootstrapping:

            values, boots = [aucROC, cutOff], [aucROCs, cutOffs]

            lowerCI = [original - 1.97 * np.std(boot) for original, boot in zip(values, boots)]
            lowerCI = [ci if ci > 0 else 0.0 for ci in lowerCI[:1]] + [lowerCI[1]]

            upperCI = [original + 1.97 * np.std(boot) for original, boot in zip(values, boots)]
            upperCI = [ci if ci < 1 else 1.0 for ci in upperCI[:1]] + [upperCI[1]]

            summary = pd.DataFrame({"Value"    : values,
                                    "CI 2.5%"  : lowerCI,
                                    "CI 97.5%" : upperCI},
                                    index = ["Area under curve ROC",
                                             "Best Cutoff"])

            results["summary"] = summary


        return results