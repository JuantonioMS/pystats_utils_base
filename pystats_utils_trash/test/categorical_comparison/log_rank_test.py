import pandas as pd
import numpy as np

from lifelines.statistics import logrank_test


from pystats_utils.test.categorical_comparison import CategoricalComparison
from pystats_utils.data_operations import reduceDataframe


class LogRankTest(CategoricalComparison):



    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 timeVariable: str = "",
                 eventVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable
        self.timeVariable = timeVariable
        self.eventVariable = eventVariable



    def cleanData(self):
        return reduceDataframe(self.dataframe,
                               self.classVariable,
                               self.timeVariable,
                               self.eventVariable)



    def cookData(self, dataframe):

        dataframe[self.eventVariable] = pd.get_dummies(dataframe[self.eventVariable],
                                                       drop_first = True)

        return dataframe



    def runTest(self, data):

        results = {"pvalue"    : {},
                   "statistic" : {},
                   "df"        : {}}

        auxData = pd.get_dummies(data[self.classVariable])

        results

        for col1 in auxData.columns:
            for col2 in auxData.columns:
                if col1 != col2:

                    title = " vs. ".join(sorted([col1, col2]))

                    res = logrank_test(data[auxData[col1] == 1][self.timeVariable],
                                       data[auxData[col2] == 1][self.timeVariable],
                                       event_observed_A = data[auxData[col1] == 1][self.eventVariable],
                                       event_observed_B = data[auxData[col2] == 1][self.eventVariable])

                    results["pvalue"][title] = res.p_value
                    results["statistic"][title] = res.test_statistic
                    results["df"][title] = res.degrees_of_freedom


        return results