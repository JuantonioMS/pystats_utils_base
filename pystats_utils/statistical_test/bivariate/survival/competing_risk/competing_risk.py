import cmprsk.cmprsk as rsk
import pandas as pd

from pystats_utils.statistical_test.bivariate.survival.survival import SurvivalTest
from pystats_utils.configuration.key_words import P_VALUE

class CompetingRiskTest(SurvivalTest):

    def __init__(self,
                 database: str = "",
                 timeVariable: str = "",
                 eventVariable: str =  "",
                 mainEvent: str = "",
                 competingEvent: str = "",
                 independentVariable: str = "") -> None:

        super().__init__(database = database,
                         timeVariable = timeVariable,
                         eventVariable = eventVariable,
                         independentVariable = independentVariable)
        self._mainEvent = mainEvent
        self._competingEvent = competingEvent



    def _preProcessData(self, registers) -> dict:

        data = super()._preProcessData(registers)

        mask = {self._mainEvent: 1,
                self._competingEvent: 2}

        data[self._eventVariable] = [mask[value] if value in mask else 0 for value in data[self._eventVariable]]

        return data



    def _runTest(self, data: dict) -> dict:

        results = {}

        for column in data.columns[2:]:

            if len(data[column].unique()) == 1: continue

            rskResult = rsk.crr(data[self._timeVariable],
                                data[self._eventVariable],
                                pd.DataFrame(pd.to_numeric(data[column])))

            results[column] = {"coef": rskResult.summary["coefficients"].values[0],
                               "se": rskResult.summary["std"].values[0],
                               P_VALUE: rskResult.summary["p_values"].values[0],
                               "hazard ratio": rskResult.summary["hazard_ratio"].values[0],
                               "hazard ratio ci lower": rskResult.summary["hazard_ratio_2.5%"].values[0],
                               "hazard ratio ci upper": rskResult.summary["hazard_ratio_97.5%"].values[0]}

        return results