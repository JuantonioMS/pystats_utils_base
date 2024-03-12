import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import resample


from pystats_utils.configuration.key_words import P_VALUE, STATISTIC
from pystats_utils.statistical_test.bivariate.value_comparison.value_comparison import ValueComparisonTest

class RocAnalysis(ValueComparisonTest):



    def run(self, bootstrap = 1000):

        self._bootstrap = bootstrap

        return super().run()



    def _preProcessData(self, registers) -> dict:

        dependentData = self._database._configuration[self._dependentVariable].variableToDataframe(registers)
        independentDate = self._database._configuration[self._independentVariable].variableToDataframe(registers)

        auxData = pd.concat([dependentData, independentDate],
                            axis = "columns",
                            join = "inner",
                            ignore_index = False,
                            verify_integrity = True)

        return auxData



    def _runTest(self, data) -> dict:

        results = {}

        for group in [column for column in data.columns if column != self._independentVariable]:

            results[group] = {}

            aucROC, cutOff = self._rocCurve(data,
                                            group,
                                            self._independentVariable)

            results[group]["aucROC"] = aucROC
            results[group]["cutOff"] = cutOff

            if self._bootstrap:

                aucROCs, cutOffs = [], []
                for _ in range(self._bootstrap):
                    bootstrapAucROC, bootstrapCutOff = self._rocCurve(resample(data,
                                                                               stratify = data[group]),
                                                                      group,
                                                                      self._independentVariable)

                    if bootstrapAucROC != float("inf") and bootstrapCutOff != float("inf"):
                        aucROCs.append(bootstrapAucROC)
                        cutOffs.append(bootstrapCutOff)

                lower, upper = self._confidenceInterval(aucROC, aucROCs)
                results[group]["aucROC CI 2.5%"] = lower if lower > 0 else 0
                results[group]["aucROC CI 97.5%"] = upper if upper < 1 else 1

                lower, upper = self._confidenceInterval(cutOff, cutOffs)
                results[group]["cutOff CI 2.5%"] = lower
                results[group]["cutOff CI 97.5%"] = upper

        return results



    @staticmethod
    def _confidenceInterval(mean, values):
        return mean - 1.97 * np.std(values), mean + 1.97 * np.std(values)



    @staticmethod
    def _rocCurve(data, dependentVariable, independentVariable):

        aucROC = roc_auc_score(data[dependentVariable],
                               data[independentVariable])

        fpr, tpr, thresholds = roc_curve(data[dependentVariable],
                                        data[independentVariable])

        info = pd.DataFrame({"FPR"       : fpr,
                             "TPR"       : tpr,
                             "Threshold" : thresholds,
                             "opt"       : tpr + (1 - fpr)})

        cutOff = info['Threshold'].iloc[info['opt'].idxmax()]

        return aucROC, cutOff