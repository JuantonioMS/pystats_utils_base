import pandas as pd
import numpy as np

from lifelines import CoxPHFitter

from pystats_utils.data_operations import isCategorical
from pystats_utils.test.multivariant import Multivariant
from pystats_utils.result import Result

class CoxPhRegression(Multivariant):


    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 eventVariable: str = "",
                 timeVariable: str = "",
                 targetVariables: list = []):

        self.dataframe = dataframe

        self.eventVariable = eventVariable

        self.timeVariable = timeVariable

        self.targetVariables = targetVariables


    def run(self) -> Result:

        """
        1. Reducir los datos de trabajo
        2. Formatear los datos para pasarlos directamente a la función de interés
        3. Correr el test
        4. Formatear los resultados
        """

        workingDataframe = self.reduceDataframe(self.dataframe,
                                                *[self.eventVariable, self.timeVariable] + self.targetVariables)

        workingData = self.extractData(workingDataframe = workingDataframe,
                                       eventVariable = self.eventVariable,
                                       timeVariable = self.timeVariable,
                                       targetVariables = self.targetVariables)

        testResults = self.runTest(workingData,
                                   eventVariable = self.eventVariable,
                                   timeVariable = self.timeVariable,
                                   targetVariables = self.targetVariables)

        return self.formatResults(**testResults)


    def extractData(self,
                    workingDataframe: pd.DataFrame = pd.DataFrame(),
                    eventVariable: str = "",
                    timeVariable: str = "",
                    targetVariables: list = []):

        for column in [eventVariable, timeVariable] + targetVariables:

            if isCategorical(workingDataframe, column):

                if len(set(list(workingDataframe[column]))) < 3:

                    if column == eventVariable:

                        aux = pd.get_dummies(workingDataframe[column],
                                             drop_first = True)

                        aux = aux.set_axis([column], axis = 1)

                    else:

                        aux = pd.get_dummies(workingDataframe[column],
                                             drop_first = True,
                                             prefix = column)

                else:

                    aux = pd.get_dummies(workingDataframe[column],
                                         drop_first = False,
                                         prefix = column)

                workingDataframe = workingDataframe.drop(column, axis = 1)

                workingDataframe = pd.concat([workingDataframe, aux],
                                              axis = 1)


        return workingDataframe


    def runTest(self,
                workingDataframe: pd.DataFrame = pd.DataFrame(),
                eventVariable: str = "",
                timeVariable: str = "",
                targetVariables: list = []) -> dict:

        cph = CoxPHFitter()
        cph.fit(workingDataframe,
                duration_col = timeVariable,
                event_col = eventVariable)

        resHR = cph.hazard_ratios_
        resCIHR = np.exp(cph.confidence_intervals_)
        resPvalues = pd.Series(cph._compute_p_values(),
                               index = list(resHR.index),
                               name = "P values")
        resParams = cph.params_
        resCI = cph.confidence_intervals_
        resError = cph.standard_errors_

        resNames = pd.Series(list(resHR.index),
                             index = list(resHR.index),
                             name = "Predictor")

        params = pd.concat([resNames,
                            resHR, resCIHR, resPvalues,
                            resParams, resCI, resError],
                            axis = 1)

        params = params.set_axis(["Predictor",
                                  "aHR", "CI 2.5%", "CI 97.5%", "P values",
                                  "Raw Coef", "Raw CI 2.5%", "Raw CI 97.5%",
                                  "Std Error"], axis = 1)

        return {"model" : cph,
                "params" : params}


    def formatResults(self, **testResults) -> Result:

        return Result(test = self.prettyName,
                      **testResults)