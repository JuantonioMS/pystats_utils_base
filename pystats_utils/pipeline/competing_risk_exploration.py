import pandas as pd

from pystats_utils.statistical_test.bivariate.survival.competing_risk.competing_risk import CompetingRiskTest
from pystats_utils.configuration.key_words import P_VALUE

class CompetingRiskExploration:



    def __init__(self,
                 database: str = "",
                 timeVariable: str = "",
                 eventVariable: str =  "",
                 mainEvent: str = "",
                 competingEvent: str = "",
                 excludedVariables: list = []) -> None:

        self._database = database
        self._timeVariable = timeVariable
        self._eventVariable = eventVariable
        self._mainEvent = mainEvent
        self._competingEvent = competingEvent
        self._excludedVariables = excludedVariables



    def run(self) -> pd.DataFrame:

        dataframe = []
        for variable in self._database.iterConfiguration():

            if not variable.type in ["integer", "float", "percentage", "ranking",
                                     "nominal", "ordinal", "binomial", "boolean",
                                     "multilabel"]: continue

            if variable.name in self._excludedVariables + [self._timeVariable, self._eventVariable]:
                continue

            rskResult = CompetingRiskTest(database = self._database,
                                          timeVariable = self._timeVariable,
                                          eventVariable = self._eventVariable,
                                          mainEvent = self._mainEvent,
                                          competingEvent = self._competingEvent,
                                          independentVariable = variable.name).run()

            if not rskResult: continue

            if variable.type in ["integer", "float", "percentage", "ranking"]:
                dataframe.append([variable.name,
                                  variable.type,
                                  "{:.3f} ({:.3f} - {:.3f})".format(rskResult[variable.name]["hazard ratio"],
                                                                    rskResult[variable.name]["hazard ratio ci lower"],
                                                                    rskResult[variable.name]["hazard ratio ci upper"]),
                                  round(rskResult[variable.name][P_VALUE], 3)])

            else:
                row = [[variable.name, variable.type, "", ""]]

                for element in rskResult:
                    row.append([element,
                                "boolean",
                                "{:.3f} ({:.3f} - {:.3f})".format(rskResult[element]["hazard ratio"],
                                                                  rskResult[element]["hazard ratio ci lower"],
                                                                  rskResult[element]["hazard ratio ci upper"]),
                                round(rskResult[element][P_VALUE], 3)])

                dataframe += row

        return pd.DataFrame(dataframe,
                            columns = ["variable",
                                       "type",
                                       "SHR (ci)",
                                       P_VALUE])