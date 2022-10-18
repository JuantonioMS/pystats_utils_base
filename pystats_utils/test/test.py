import pandas as pd

from pystats_utils.result import Result

class Test:

    alpha = 0.05

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariable = targetVariable


    def run(self) -> Result:

        """
        1. Reducir los datos de trabajo
        2. Formatear los datos para pasarlos directamente a la función de interés
        3. Correr el test
        4. Formatear los resultados
        """

        workingDataframe = self.reduceDataframe(self.dataframe,
                                                *[column for column in [self.classVariable,
                                                                        self.targetVariable] if column])

        workingData = self.extractData(workingDataframe = workingDataframe,
                                       classVariable = self.classVariable,
                                       targetVariable = self.targetVariable)

        testResults = self.runTest(workingData)

        return self.formatResults(**testResults)


    #  ____________________Reduce Dataframe____________________


    def reduceDataframe(self,
                        dataframe: pd.DataFrame,
                        *columns) -> pd.DataFrame:

        from pystats_utils.data_operations import  reduceDataframe

        return reduceDataframe(dataframe, *columns)


    #  ____________________Extract Data____________________


    def extractData(self,
                    workingDataframe: pd.DataFrame = pd.DataFrame(),
                    classVariable: str = "",
                    targetVariable: str = "") -> list:

        if self.isMonovariant:

            return self.extractDataMonovariant(workingDataframe = workingDataframe,
                                               targetVariable = targetVariable)

        else:

            return self.extractDataBivariant(workingDataframe = workingDataframe,
                                             classVariable = classVariable,
                                             targetVariable = targetVariable)


    def extractDataMonovariant(self,
                               workingDataframe: pd.DataFrame = pd.DataFrame(),
                               targetVariable: str = "") -> list:

        return [list(workingDataframe[targetVariable])]


    def extractDataBivariant(self,
                             workingDataframe: pd.DataFrame = pd.DataFrame(),
                             classVariable: str = "",
                             targetVariable: str = "") -> list:

        data = []

        classes = set(list(workingDataframe[classVariable]))

        for clas in classes:

            data.append(list(workingDataframe[workingDataframe[classVariable] == clas][targetVariable]))

        return data


    #  ____________________Run Test____________________


    def runTest(self,
                workingData: list) -> dict: return {}


    #  ____________________Format Result____________________


    def formatResults(self,
                      **testResults) -> Result:

        return Result(test = self.prettyName,
                      significance = testResults["pvalue"] < self.alpha,
                      **testResults)


    #  ____________________Getters____________________


    @property
    def prettyName(self):
        return "".join([f" {char}" if char.isupper() else char for char in self.__class__.__name__]).strip(" ")


    @property
    def context(self) -> str:
        return self.__class__.__base__.__name__.capitalize()


    @property
    def isMonovariant(self) -> bool:
        return not bool(self.classVariable)


    @property
    def isBivariant(self) -> bool:
        return not self.isMonovariant