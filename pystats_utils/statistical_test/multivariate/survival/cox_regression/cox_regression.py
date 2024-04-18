import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)

from pystats_utils.statistical_test.multivariate.survival.survival import SurvivalTest
from pystats_utils.configuration.key_words import P_VALUE

class CoxRegressionTest(SurvivalTest):



    def _runTest(self, data: pd.DataFrame) -> dict:

        results = {}

        try:

            cph = CoxPHFitter()
            cph.fit(data,
                    duration_col = self._timeVariable,
                    event_col = self._eventVariable)

        except ConvergenceError: return results

        results["params"] = pd.concat([cph.params_,
                                       cph.standard_errors_,
                                       cph.hazard_ratios_,
                                       np.exp(cph.confidence_intervals_),
                                       pd.Series(cph._compute_p_values(),
                                                 name = P_VALUE,
                                                 index = [column \
                                                          for column in data.columns \
                                                          if not column in [self._timeVariable, self._eventVariable]])],
                                      axis = "columns")

        results["summary"] = {"concordance": cph.concordance_index_,
                              "partial AIC": cph.AIC_partial_,
                              "partial log-likelihood": cph.log_likelihood_,
                              "log-likelihood ratio statistic": cph.log_likelihood_ratio_test().test_statistic,
                              "log-likelihood ratio df": cph.log_likelihood_ratio_test().degrees_freedom,
                              "log-likelihood ratio p value": cph.log_likelihood_ratio_test().p_value,
                              "log-likelihood ratio -log2(p)": cph.log_likelihood_ratio_test().summary["-log2(p)"][0]}

        return results