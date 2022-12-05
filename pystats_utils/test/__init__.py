import pandas as pd

from pystats_utils.result import Result
from pystats_utils.data_operations import  reduceDataframe


class Test:


    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(), #  Dataframe con todos los datos
                 classVariable: str = "",                  #  Grupos a comparar
                 targetVariable: str = ""):                #  Variable a comparar

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariable = targetVariable


    def run(self) -> Result:

        """
        Paso 1. Eliminar variables no indicadas y quitar registros con datos faltantes
        Paso 2. Formatear los datos ajustados para el test
        Paso 3. Correr el test
        Paso 4. Formatear los resultados
        """

        auxDataframe = self.cleanData()

        auxData = self.cookData(auxDataframe)

        testResult = self.runTest(auxData)

        return self.cookResult(**testResult)


    def cleanData(self):

        columns = [self.classVariable] if self.classVariable else []

        if not isinstance(self.targetVariable, str):
            for variable in self.targetVariable: columns.append(variable)

        else:
            columns.append(self.targetVariable)

        return reduceDataframe(self.dataframe, *columns)


    def cookData(self, dataframe):

        if self.classVariable:  #  Es bivariante

            data = {}

            for group in set(list(dataframe[self.classVariable])):

                data[group] = list(dataframe[dataframe[self.classVariable] == group][self.targetVariable])

            return data

        else:  #  Es monovariante

            return list(dataframe[self.targetVariable])


    def runTest(self, data):

        return {}


    def cookResult(self, **result):

        fields = ["pvalue", "statistic"]

        for field in fields:

            if field in result:

                if not isinstance(result[field], float) and len(result[field]) < 2:
                    result[field] = list(result[field].values())[0]

        return Result(test = self.prettyName,
                      context = self.context,
                      **result)


    @property
    def prettyName(self):
        return "".join([f" {char}" if char.isupper() else char for char in self.__class__.__name__]).strip(" ")


    @property
    def context(self) -> str:
        return self.__class__.__base__.__name__.capitalize()