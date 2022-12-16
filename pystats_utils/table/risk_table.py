import numpy as np
import pandas as pd


from pystats_utils.table import Table
from pystats_utils.data_operations import reduceDataframe

class RiskTable(Table):

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 eventVariable: str = "",
                 timeVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.eventVariable = eventVariable

        self.timeVariable = timeVariable



    def cleanData(self):

        return reduceDataframe(self.dataframe,
                               self.classVariable,
                               self.timeVariable,
                               self.eventVariable)



    def cookData(self, dataframe):

        dataframe[self.eventVariable] = pd.get_dummies(dataframe[self.eventVariable],
                                                       drop_first = True)

        return dataframe



    def runTable(self, data, **kwargs):




        info = pd.DataFrame({"Timeline"   : list(range(np.max(data[self.timeVariable]) + 1))})

        aux = []
        for day in info["Timeline"]:

            atRisk = len(data) - np.sum(data[data[self.timeVariable] <= day][self.eventVariable])
            aux.append(atRisk)

        info["Global"] = aux

        if self.classVariable:

            for group in sorted(data[self.classVariable].unique()):

                workData = data[data[self.classVariable] == group]

                aux = []
                for day in info["Timeline"]:

                    atRisk = len(workData) - np.sum(workData[workData[self.timeVariable] <= day][self.eventVariable])
                    aux.append(atRisk)

                info[f"{self.classVariable}_{group}"] = aux


        return info