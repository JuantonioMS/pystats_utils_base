import pandas as pd

from pystats_utils.statistical_test.multivariate.multivariate import MultivariateStatisticalTest

class SurvivalTest(MultivariateStatisticalTest):



    def __init__(self,
                 database: str = "",
                 timeVariable: str = "",
                 eventVariable: str =  "",
                 independentVariables: list = []) -> None:

        self._database = database
        self._timeVariable = timeVariable
        self._eventVariable = eventVariable
        self._independentVariables = independentVariables



    def _mergeVariables(self) -> list:

        return [self._timeVariable, self._eventVariable] + self._independentVariables



    def _preProcessData(self, registers):

        data = pd.DataFrame([[register[self._eventVariable], register[self._timeVariable]] for register in registers],
                            index = [register.id for register in registers],
                            columns = [self._eventVariable,
                                       self._timeVariable])

        for variable in self._independentVariables:
            variable = self._database._configuration[variable]

            auxData = variable.variableToDataframe(registers)

            if variable.type in ["nominal", "ordinal", "binomial", "boolean"]:
                auxData = auxData[auxData.columns[1:]]

            data = pd.concat([data, auxData],
                             axis = "columns")

        return data