import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import statsmodels.formula.api as smf

import pandas as pd
import numpy as np

from pystats_utils.statistical_test.bivariate.bivariate import BivariateStatisticalTest

class LogisticRegression(BivariateStatisticalTest):



    def _preProcessData(self, registers: list) -> pd.DataFrame:

        data = pd.concat([self._database._configuration[self._dependentVariable].variableToDataframe(registers),
                          self._database._configuration[self._independentVariable].variableToDataframe(registers)],
                         axis = "columns",
                         ignore_index = False)

        for column in data.columns:
            data[column] = data[column].astype(float)

        return data



    def _runTest(self, data: pd.DataFrame) -> dict:

        data.columns = [column.replace(".", "_") for column in data.columns]

        results = {}
        for column in data.columns:

            if self._dependentVariable in column: continue
            else: results[column] = {}

            data.to_excel("test.xlsx")

            formula = f"{data.columns[1]} ~ {column}"
            model = smf.logit(formula = formula,
                              data = data).fit(disp = 0)

            results[column]["formula"] = formula
            results[column]["model"] = model