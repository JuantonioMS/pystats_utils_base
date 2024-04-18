import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)

from pystats_utils.statistical_test.bivariate.survival.survival import SurvivalTest
from pystats_utils.configuration.key_words import P_VALUE

class CoxRegressionTest(SurvivalTest):



    def _runTest(self, data: pd.DataFrame) -> dict:

        results = {}

        for column in data.columns[2:]:

            if len(data[column].unique()) == 1: continue

            try:
                cph = CoxPHFitter()
                cph.fit(data[[self._timeVariable, self._eventVariable, column]],
                        duration_col = self._timeVariable,
                        event_col = self._eventVariable)

            except ConvergenceError:
                continue

            results[column] = {}
            results[column]["model"] = cph

            results[column]["hazard ratio"] = cph.hazard_ratios_.values[0]
            results[column]["hazard ratio ci lower"] = np.exp(cph.confidence_intervals_).values[0][0]
            results[column]["hazard ratio ci upper"] = np.exp(cph.confidence_intervals_).values[0][1]

            results[column][P_VALUE] = cph._compute_p_values()[0]

            results[column]["coef"] = cph.params_.values[0]
            results[column]["coef ci lower"] = cph.confidence_intervals_.values[0][0]
            results[column]["coef ci upper"] = cph.confidence_intervals_.values[0][1]
            results[column]["se"] = cph.standard_errors_.values[0]

            results[column]["summary"] = {"concordance": cph.concordance_index_,
                                          "partial AIC": cph.AIC_partial_,
                                          "partial log-likelihood": cph.log_likelihood_,
                                          "log-likelihood ratio statistic": cph.log_likelihood_ratio_test().test_statistic,
                                          "log-likelihood ratio df": cph.log_likelihood_ratio_test().degrees_freedom,
                                          "log-likelihood ratio p value": cph.log_likelihood_ratio_test().p_value,
                                          "log-likelihood ratio -log2(p)": cph.log_likelihood_ratio_test().summary["-log2(p)"][0]}

        return results


