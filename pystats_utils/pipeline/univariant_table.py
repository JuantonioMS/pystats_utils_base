import pandas as pd
import numpy as np

from pystats_utils.data_operations import isCategorical
from pystats_utils.data_operations import reduceDataframe


class UnivariantTable:

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 excludedVariables: list = []):

        self.dataframe = dataframe

        self.excludedVariables = excludedVariables


    def run(self) -> pd.DataFrame:

        header = {"Variable" : pd.Series(dtype = "str"),
                  "Information" : pd.Series(dtype = "str"),
                  "Non empty" : pd.Series(dtype = "int")}


        table = pd.DataFrame(header)

        template = {col : [] for col in table}
        for column in self.dataframe:

            if column in self.excludedVariables: continue

            workDataframe = reduceDataframe(self.dataframe,
                                            column)

            template["Variable"].append(column)
            template["Non empty"].append(len(workDataframe))

            if isCategorical(workDataframe, column):

                template["Information"].append("")

                aux = pd.get_dummies(workDataframe[column],
                                     prefix = column)

                for auxColumn in aux:

                    template["Variable"].append(f"----> {auxColumn}")
                    template["Information"].append("{} ({:.2f})".format(np.sum(aux[auxColumn]),
                                                                        np.sum(aux[auxColumn]) /\
                                                                        len(aux[auxColumn]) * 100))
                    template["Non empty"].append("")

            else:
                template["Information"].append("{:.2f} ({:.2f} - {:.2f})".format(np.mean(workDataframe[column]),
                                                                                 np.percentile(workDataframe[column], 25),
                                                                                 np.percentile(workDataframe[column], 75)))


        table = pd.concat([table, pd.DataFrame(template)])

        return table