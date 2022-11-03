from sys import prefix
import pandas as pd

from pystats_utils.test import Test
from pystats_utils.result import Result

from pystats_utils.data_operations import isCategorical
class Multivariant(Test):

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariables: list = []):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariables = targetVariables


    def run(self) -> Result:

        """
        1. Reducir los datos de trabajo
        2. Formatear los datos para pasarlos directamente a la función de interés
        3. Correr el test
        4. Formatear los resultados
        """

        workingDataframe = self.reduceDataframe(self.dataframe,
                                                *[self.classVariable] + self.targetVariables)

        workingData, formula = self.extractData(workingDataframe = workingDataframe,
                                                classVariable = self.classVariable,
                                                targetVariables = self.targetVariables)

        testResults = self.runTest(workingData, formula)

        return self.formatResults(**testResults)


    def extractData(self,
                    workingDataframe: pd.DataFrame = pd.DataFrame(),
                    classVariable: str = "",
                    targetVariables: list = []):

        workingDataframe[classVariable] = pd.get_dummies(workingDataframe[classVariable],
                                                         drop_first = True,
                                                         prefix = classVariable)

        for column in targetVariables:

            if isCategorical(workingDataframe, column):



                auxColumns = pd.get_dummies(workingDataframe[column],
                                            drop_first = True,
                                            prefix = column)

                workingDataframe = workingDataframe.drop(columns = column)

                workingDataframe = pd.concat([workingDataframe, auxColumns],
                                             axis = 1)

        aux = []
        for column in workingDataframe:

            if column != classVariable:
                aux.append(column)

        formula = f"{classVariable} ~ " + " + ".join(aux)

        return workingDataframe, formula


    def runTest(self,
                workingData: pd.DataFrame,
                formula: str) -> dict:

        return {}


    def formatResults(self, **testResults) -> Result:

        return Result(test = self.prettyName,
                      **testResults)

from pystats_utils.test.multivariant.logistic_regression import LogisticRegression
from pystats_utils.test.multivariant.coxph_regression import CoxPhRegression