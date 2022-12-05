from sys import prefix
import pandas as pd

from pystats_utils.test import Test
from pystats_utils.result import Result

from pystats_utils.data_operations import isCategorical
class MultivariantTest(Test):

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariable: list = []):

        super().__init__(dataframe = dataframe,
                         classVariable = classVariable,
                         targetVariable = targetVariable)



    def cookData(self, dataframe):

        dataframe[self.classVariable] = pd.get_dummies(dataframe[self.classVariable],
                                                       drop_first = True)

        for column in self.targetVariable:

            if isCategorical(dataframe, column):

                auxColumns = pd.get_dummies(dataframe[column],
                                            drop_first = True,
                                            prefix = column)

                dataframe = dataframe.drop(columns = column)

                dataframe = pd.concat([dataframe, auxColumns], axis = 1)

        return dataframe



    def formatResults(self, **testResults) -> Result:

        return Result(test = self.prettyName,
                      **testResults)

from pystats_utils.test.multivariant.logistic_regression import LogisticRegression
#from pystats_utils.test.multivariant.coxph_regression import CoxPhRegression