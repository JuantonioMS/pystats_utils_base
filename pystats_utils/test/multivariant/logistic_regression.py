import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

from pystats_utils.test.multivariant import Multivariant

class LogisticRegression(Multivariant):

    def runTest(self,
                workingData: pd.Dataframe,
                formula: str) -> dict:

        model = smf.logit(formula, data = workingData).fit()

        # Resutl

        resCoef = model.params
        resOr = np.exp(resCoef)

        resPvalue = model.pvalues

        resErr = model.bse
        resConf = model.conf_int(0.05)
        resConfOr = np.exp(resConf)

        result = pd.concat([resOr, resConfOr, resPvalue,
                            resCoef, resConf,
                            resErr], axis = 1)
        
        result = result.reset_index()
        
        result = result.set_axis([])
        
        res = res.set_axis(["A", "B", "c", "d", "f", "g", "h", "i"], axis = 1)
        res = res.reset_index()


        return {"summary" : model}