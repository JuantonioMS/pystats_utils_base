import pandas as pd
import numpy as np

from pystats_utils.test.multivariant import CoxPhRegression

from pystats_utils.data_operations import isCategorical
from pystats_utils.data_operations import reduceDataframe


class CoxExploration:

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 eventVariable: str = "",
                 timeVariable: str = "",
                 excludedVariables: list = []):

        self.dataframe = dataframe

        self.eventVariable = eventVariable
        
        self.timeVariable = timeVariable

        self.excludedVariables = excludedVariables


    def run(self) -> pd.DataFrame:

        header = {"Variable" : pd.Series(dtype = "str"),
                  "HR(95CI)" : pd.Series(dtype = "str"),
                  "P value"  : pd.Series(dtype = "float")}

        table = pd.DataFrame(header)

        for column in self.dataframe:

            if column not in self.excludedVariables + [self.eventVariable, self.timeVariable]:

                try:

                    workDataframe = reduceDataframe(self.dataframe,
                                                    self.eventVariable,
                                                    self.timeVariable,
                                                    column)

                    if len(set(workDataframe[column])) == 1:
                        continue

                    if len(set(workDataframe[column])) > 2 and isCategorical(workDataframe, column):

                        auxColumns = pd.get_dummies(workDataframe[column],
                                                    prefix = column)

                        workDataframe = workDataframe.drop(columns = [column])

                        workDataframe = pd.concat([workDataframe, auxColumns],
                                                  axis = 1)


                    template = dict([(key, [""]) for key in header])

                    for targetVariable in [targetVariable for targetVariable in workDataframe.columns if targetVariable not in [self.eventVariable,
                                                                                                                                self.timeVariable]]:

                        result = CoxPhRegression(dataframe = workDataframe,
                                                 eventVariable = self.eventVariable,
                                                 timeVariable = self.timeVariable,
                                                 targetVariables = [targetVariable]).run()

                        result.params["HR(95CI)"] = result.params.apply(lambda row: formatCell(row), axis = 1)
                        result.params["P value"] = result.params.apply(lambda row: round(row["P values"], 3), axis = 1)

                        for index, row in result.params.iterrows():

                            if index != 0:
                                template["Variable"].append(row["Predictor"])
                                template["HR(95CI)"].append(row["HR(95CI)"])
                                template["P value"].append(row["P value"])

                    for key in template:
                        template[key] = template[key][1:]

                    table = pd.concat([table,
                                    pd.DataFrame(template)])

                except:
                    pass

        return table


def formatCell(row):

    cell = f"{round(row['aHR'], 2)} ({round(row['CI 2.5%'], 2)} - {round(row['CI 97.5%'], 2)})"

    return cell