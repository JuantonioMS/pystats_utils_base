import pandas as pd

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
                  "P value"  : pd.Series(dtype = "str")}

        table = pd.DataFrame(header)

        template = dict([(key, []) for key in header])
        for column in self.dataframe:

            if column not in self.excludedVariables + [self.eventVariable, self.timeVariable]:

                try:

                    workDataframe = reduceDataframe(self.dataframe,
                                                    self.eventVariable,
                                                    self.timeVariable,
                                                    column)

                    if isCategorical(workDataframe, column):

                         auxDataframe = pd.get_dummies(workDataframe[column],
                                                       prefix = column)

                         for auxColumn in auxDataframe:

                            result = CoxPhRegression(dataframe = pd.concat([workDataframe[self.eventVariable],
                                                                            workDataframe[self.timeVariable],
                                                                            auxDataframe[auxColumn]],
                                                                            axis = 1),
                                                     eventVariable = self.eventVariable,
                                                     timeVariable = self.timeVariable,
                                                     targetVariable = [auxColumn]).run()

                            for _, row in result.params.iterrows():

                                template["Variable"].append(row["Predictor"])
                                template["HR(95CI)"].append("{:.2f} ({:.2f} - {:.2f}".format(row["aHR"],
                                                                                             row["CI 2.5%"],
                                                                                             row["CI 97.5%"]))
                                template["P value"].append("{:.3f}".format(row["P values"]))

                    else:

                        result = CoxPhRegression(dataframe = workDataframe,
                                                 eventVariable = self.eventVariable,
                                                 timeVariable = self.timeVariable,
                                                 targetVariable = [column]).run()

                        for _, row in result.params.iterrows():

                            template["Variable"].append(row["Predictor"])
                            template["HR(95CI)"].append("{:.2f} ({:.2f} - {:.2f}".format(row["aHR"],
                                                                                         row["CI 2.5%"],
                                                                                         row["CI 97.5%"]))
                            template["P value"].append("{:.3f}".format(row["P values"]))

                except:
                    pass

        table = pd.concat([table, pd.DataFrame(template)])

        return table