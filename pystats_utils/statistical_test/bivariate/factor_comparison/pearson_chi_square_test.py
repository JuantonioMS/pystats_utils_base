from scipy.stats import chi2_contingency
from scipy.stats.contingency import crosstab
import pandas as pd

from pystats_utils.statistical_test.bivariate.factor_comparison.factor_comparison import FactorComparison
from pystats_utils.configuration.key_words import P_VALUE, STATISTIC, DOF, EXPECTED


class PearsonChiSquareTest(FactorComparison):



    def _preProcessData(self, registers) -> dict:

        if self._database._configuration[self._independentVariable].type in ("nominal", "ordinal"):

            data = pd.DataFrame(data = [[register[self._dependentVariable],
                                         register[self._independentVariable]] \
                                        for register in registers],
                                index = [register.id for register in registers],
                                columns = [self._dependentVariable,
                                           self._independentVariable])

            data = pd.concat([data,
                              self._database._configuration[self._independentVariable].variableToDataframe(registers)],
                             axis = "columns",
                             join = "inner",
                             ignore_index = False,
                             verify_integrity = True)

        elif self._database._configuration[self._independentVariable].type in ("binomial", "boolean"):

            data = pd.DataFrame(data = [[register[self._dependentVariable],
                                         register[self._independentVariable]] \
                                        for register in registers],
                                index = [register.id for register in registers],
                                columns = [self._dependentVariable,
                                           self._independentVariable])

        elif self._database._configuration[self._independentVariable].type == "multilabel":

            data = pd.DataFrame(data = [register[self._dependentVariable] for register in registers],
                                index = [register.id for register in registers],
                                columns = [self._dependentVariable])

            data = pd.concat([data,
                              self._database._configuration[self._independentVariable].variableToDataframe(registers)],
                             axis = "columns",
                             join = "inner",
                             ignore_index = False,
                             verify_integrity = True)

        return data



    def _runTest(self, data: pd.DataFrame) -> dict:

        results = {}

        if self._independentVariable in data.columns:

            statistic, pValue, dof, expected = self._coreFunction(data,
                                                                  self._dependentVariable,
                                                                  self._independentVariable)

            results["all"] = {STATISTIC : statistic,
                              P_VALUE : pValue,
                              DOF : dof,
                              EXPECTED : expected}

        if len(data.columns) > 2:

            for column in data.columns:
                if not column in [self._dependentVariable,
                                  self._independentVariable]:

                    statistic, pValue, dof, expected = self._coreFunction(data,
                                                                          self._dependentVariable,
                                                                          column)

                    results[column] = {STATISTIC : statistic,
                                       P_VALUE : pValue,
                                       DOF : dof,
                                       EXPECTED : expected}

        return results



    @staticmethod
    def _coreFunction(data: pd.DataFrame,
                      dependentVariable: str,
                      independentVariable: str):

        _, contigenceTab = crosstab(list(data[dependentVariable]),
                                    list(data[independentVariable]))

        statistic, pValue, dof, expected = chi2_contingency(contigenceTab)

        return statistic, pValue, dof, expected