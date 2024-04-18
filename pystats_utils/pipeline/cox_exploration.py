import pandas as pd

from pystats_utils.statistical_test.bivariate.survival.cox_regression.cox_regression import CoxRegressionTest
from pystats_utils.configuration.key_words import P_VALUE

class CoxExploration:

    def __init__(self,
                 database: str = "",
                 timeVariable: str = "",
                 eventVariable: str =  "",
                 excludedVariables: list = []) -> None:

        self._database = database
        self._timeVariable = timeVariable
        self._eventVariable = eventVariable
        self._excludedVariables = excludedVariables



    def run(self) -> pd.DataFrame:

        dataframe = []
        for variable in self._database.iterConfiguration():

            if not variable.type in ["integer", "float", "percentage", "ranking",
                                     "nominal", "ordinal", "binomial", "boolean",
                                     "multilabel"]: continue

            if variable.name in self._excludedVariables + [self._timeVariable, self._eventVariable]:
                continue

            coxResults = CoxRegressionTest(database = self._database,
                                           timeVariable = self._timeVariable,
                                           eventVariable = self._eventVariable,
                                           independentVariable = variable.name).run()

            if not coxResults: continue

            if variable.type in ["integer", "float", "percentage", "ranking"]:
                dataframe.append([variable.name,
                                  variable.type,
                                  "{:.3f} ({:.3f} - {:.3f})".format(coxResults[variable.name]["hazard ratio"],
                                                                    coxResults[variable.name]["hazard ratio ci lower"],
                                                                    coxResults[variable.name]["hazard ratio ci upper"]),
                                  round(coxResults[variable.name][P_VALUE], 3)])

            else:
                row = [[variable.name, variable.type, "", ""]]

                for element in coxResults:
                    row.append([element,
                                "boolean",
                                "{:.3f} ({:.3f} - {:.3f})".format(coxResults[element]["hazard ratio"],
                                                                  coxResults[element]["hazard ratio ci lower"],
                                                                  coxResults[element]["hazard ratio ci upper"]),
                                round(coxResults[element][P_VALUE], 3)])

                dataframe += row

        return pd.DataFrame(dataframe,
                            columns = ["variable",
                                       "type",
                                       "hazard ratio (ci)",
                                       P_VALUE])