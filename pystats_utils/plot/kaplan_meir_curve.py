
import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
import seaborn as sns

from pystats_utils.plot import Plot
from pystats_utils.data_operations import reduceDataframe

class RidgePlot(Plot):

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariable = targetVariable



    def cleanData(self):

        return reduceDataframe(self.dataframe,
                               self.classVariable,
                               self.targetVariable)



    def cookData(self, dataframe):
        return dataframe
