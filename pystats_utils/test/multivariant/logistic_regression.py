import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import statsmodels.formula.api as smf

import pandas as pd
import numpy as np

from pystats_utils.test.multivariant import MultivariantTest
from pystats_utils.test.value_comparison import RocAnalysis
from pystats_utils.test.categorical_comparison import ConfussionMatrixAnalysis

class LogisticRegression(MultivariantTest):


    def runTest(self, data):

        results = {}

        columns = [column for column in data.columns if column != self.classVariable]

        formula = f"{self.classVariable}" + " ~ " + " + ".join(columns)

        model = smf.logit(formula, data = data).fit(disp = 0)

        results["model"] = model

        #  Result
        resCoef = model.params
        resOr = np.exp(resCoef)
        resPvalue = model.pvalues
        resErr = model.bse
        resConf = model.conf_int(0.05)
        resConfOr = np.exp(resConf)

        params = pd.concat([resOr, resConfOr, resPvalue,
                            resCoef, resConf,
                            resErr], axis = 1)

        params = params.reset_index()

        params = params.set_axis(["Predictor",
                                  "aOR", "CI 2.5%", "CI 97.5%", "P value",
                                  "Coef", "Raw CI 2.5%", "Raw CI 97.5%",
                                  "Standard Err."], axis = 1)

        results["params"] = params

        #  ROC análisis
        rocResult = RocAnalysis(dataframe = pd.DataFrame({"expected" : data[self.classVariable],
                                                          "probability" : model.predict()}),
                                classVariable = "expected",
                                targetVariable = "probability").run(bootstrapping = self.bootstrapping)

        #  Confussion matrix
        predictedClass = [1 if probability > rocResult.cutOff else 0 for probability in model.predict()]
        matrixResult = ConfussionMatrixAnalysis(dataframe = pd.DataFrame({"expected" : data[self.classVariable],
                                                             "predicted" : predictedClass}),
                                                classVariable = "expected",
                                                targetVariable = "predicted").run(bootstrapping = self.bootstrapping)

        for aux in [rocResult, matrixResult]:

            for atr in aux.__dict__:
                    if atr not in ["summary", "test", "context"]:
                            results[atr] = aux.__dict__[atr]

        if self.bootstrapping:
            results["summary"] = pd.concat((rocResult.summary,
                                            matrixResult.summary),
                                        axis = 0)

        return results