import pandas as pd

from pystats_utils.statistical_test.bivariate.bivariate import BivariateStatisticalTest

class SurvivalTest(BivariateStatisticalTest):



    def __init__(self,
                 database: str = "",
                 timeVariable: str = "",
                 eventVariable: str =  "",
                 independentVariable: str = "") -> None:

        self._database = database
        self._timeVariable = timeVariable
        self._eventVariable = eventVariable
        self._independentVariable = independentVariable



    def _preProcessData(self, registers) -> dict:

        data = pd.DataFrame([[register[self._eventVariable], register[self._timeVariable]] for register in registers],
                            index = [register.id for register in registers],
                            columns = [self._eventVariable,
                                       self._timeVariable])

        data = pd.concat([data,
                          self._database._configuration[self._independentVariable].variableToDataframe(registers)],
                         axis = "columns")

        return data



    def _mergeVariables(self) -> list:

        return [self._timeVariable,
                self._eventVariable,
                self._independentVariable]