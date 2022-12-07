import pandas as pd
import numpy as np

from lifelines import CoxPHFitter

from pystats_utils.data_operations import isCategorical, reduceDataframe
from pystats_utils.test.multivariant import MultivariantTest

class CoxPhRegression(MultivariantTest):


    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 eventVariable: str = "",
                 timeVariable: str = "",
                 targetVariable: list = []):

        self.dataframe = dataframe

        self.eventVariable = eventVariable

        self.timeVariable = timeVariable

        self.targetVariable = targetVariable


    def cleanData(self):

        return reduceDataframe(self.dataframe,
                               self.eventVariable,
                               self.timeVariable,
                               *self.targetVariable)



    def cookData(self, dataframe):

        for column in [self.eventVariable] + self.targetVariable:

            if isCategorical(dataframe, column):

                aux = pd.get_dummies(dataframe[column],
                                     drop_first = True,
                                     prefix = column)

                if column == self.eventVariable:
                    self.eventVariable = aux.columns[0]

                dataframe = dataframe.drop(column, axis = 1)
                dataframe = pd.concat([dataframe, aux], axis = 1)

        return dataframe



    def runTest(self, data):

        results = {}


        cph = CoxPHFitter()
        cph.fit(data,
                duration_col = self.timeVariable,
                event_col = self.eventVariable)

        results["model"] = cph

        #  Params
        resHR = cph.hazard_ratios_
        resCIHR = np.exp(cph.confidence_intervals_)
        resPvalues = pd.Series(cph._compute_p_values(),
                               index = list(resHR.index),
                               name = "P values")
        resParams = cph.params_
        resCI = cph.confidence_intervals_
        resError = cph.standard_errors_

        resNames = pd.Series(list(resHR.index),
                             index = list(resHR.index),
                             name = "Predictor")

        params = pd.concat([resNames,
                            resHR, resCIHR, resPvalues,
                            resParams, resCI, resError],
                            axis = 1)

        params = params.set_axis(["Predictor",
                                  "aHR", "CI 2.5%", "CI 97.5%", "P values",
                                  "Raw Coef", "Raw CI 2.5%", "Raw CI 97.5%",
                                  "Std Error"], axis = 1)

        results["params"] = params

        #  Summary
        summary = pd.DataFrame({"Values" : [cph.concordance_index_,
                                            cph.AIC_partial_,
                                            cph.log_likelihood_,
                                            cph.log_likelihood_ratio_test().test_statistic,
                                            cph.log_likelihood_ratio_test().degrees_freedom,
                                            cph.log_likelihood_ratio_test().p_value,
                                            cph.log_likelihood_ratio_test().summary["-log2(p)"][0]]},
                               index = ["Concordance",
                                        "Partial AIC",
                                        "Partial log-likelihood",
                                        "Log-likelihood ratio statistic",
                                        "Log-likelihood ratio df",
                                        "Log-likelihood ratio p value",
                                        "Log-likelihood ratio -log2(p)"])

        results["summary"] = summary

        return results