from plotnine import aes

import pandas as pd

from pystats_utils.plot import Plot

class RocCurve(Plot):


    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 expectedVariable: str = "",
                 predictedVariable: str = ""):

        self.dataframe = dataframe

        self.expectedVariable = expectedVariable

        self.predictedVariable = predictedVariable


    def run(self):

        pass