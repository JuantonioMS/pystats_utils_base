import pandas as pd
import numpy as np

from pystats_utils.test.multivariant import LogisticRegression

from pystats_utils.data_operations import isCategorical
from pystats_utils.data_operations import reduceDataframe


class LogisticExploration:

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 excludedVariables: list = []):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.excludedVariables = excludedVariables


    def run(self) -> pd.DataFrame:

        header = {"Variable" : pd.Series(dtype = "str"),
                  "OR(95CI)" : pd.Series(dtype = "str"),
                  "P value"  : pd.Series(dtype = "str")}

        table = pd.DataFrame(header)

        template = dict([(key, []) for key in header])

        for column in self.dataframe:

            if column not in self.excludedVariables + [self.classVariable]:

                    workDataframe = reduceDataframe(self.dataframe,
                                                    self.classVariable, column)

                    #  Categorical section
                    if isCategorical(workDataframe, column):

                        auxDataframe = pd.get_dummies(workDataframe[column],
                                                      prefix = column)

                        for auxColumn in auxDataframe:

                            try:

                                result = LogisticRegression(dataframe = pd.concat([workDataframe[self.classVariable],
                                                                                auxDataframe[auxColumn]],
                                                                                axis = 1),
                                                            classVariable = self.classVariable,
                                                            targetVariable = [auxColumn],
                                                            bootstrapping = 0).run()

                                for index, row in result.params.iterrows():

                                    if index != 0:
                                        template["Variable"].append(row["Predictor"])
                                        template["OR(95CI)"].append("{:.2f} ({:.2f} - {:.2f})".format(row["aOR"],
                                                                                                    row["CI 2.5%"],
                                                                                                    row["CI 97.5%"]))
                                        template["P value"].append("{:.3f}".format(row["P value"]))

                            except Exception as e:
                                print(auxColumn, e)




                    #  Numerical section
                    else:

                        try:

                            result = LogisticRegression(dataframe = workDataframe,
                                                        classVariable = self.classVariable,
                                                        targetVariable = [column],
                                                        bootstrapping = 0).run()

                            for index, row in result.params.iterrows():

                                if index != 0:
                                    template["Variable"].append(row["Predictor"])
                                    template["OR(95CI)"].append("{:.2f} ({:.2f} - {:.2f})".format(row["aOR"],
                                                                                                    row["CI 2.5%"],
                                                                                                    row["CI 97.5%"]))
                                    template["P value"].append("{:.3f}".format(row["P value"]))

                        except Exception as e:
                            print(column, e)


        table = pd.concat([table, pd.DataFrame(template)])

        return table