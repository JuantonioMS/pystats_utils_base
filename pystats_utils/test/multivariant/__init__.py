import panda as pd

from pystats_utils.test import Test
from pystats_utils.result import Result

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

        testResults = self.runTest(workingData,)

        return self.formatResults(**testResults)


    def extractData(self,
                    workingDataframe: pd.DataFrame = pd.DataFrame(),
                    classVariable: str = "",
                    targetVariables: list = []):

        formula = f"{classVariable} ~ " + " + ".join(targetVariables)

        return workingDataframe, formula


    def runTest(self,
                workingData: pd.Dataframe,
                formula: str) -> dict:

        return {}


    def formatResults(self, **testResults) -> Result:

        return Result(test = self.prettyName,
                      **testResults)
