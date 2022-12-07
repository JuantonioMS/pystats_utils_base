import pandas as pd
import numpy as np

from pystats_utils.test.categorical_comparison import PearsonChiSquareTest

from pystats_utils.data_operations import reduceDataframe

class ExtendedContingencyTable:

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariable: str = "",
                 cohortVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariable = targetVariable

        self.cohortVariable = cohortVariable



    def run(self):
        pass

        startColumns = {f"Cohort-{self.cohortVariable}" : pd.Series(dtype = "str"),
                        "All" : pd.Series(dtype = str)}

        groups = sorted(list(set(list(self.dataframe[self.classVariable].dropna()))))


        groupColumns = dict([(f"{self.classVariable} {group}", pd.Series(dtype = "str")) for group in groups])

        endColumns = {"P value" : pd.Series(dtype = "str")}

        header = {}
        header.update(startColumns)
        header.update(groupColumns)
        header.update(endColumns)

        table = pd.DataFrame(header)

        template = dict([(key, []) for key in header])
        for cohort in ["All"] + sorted(list(set(self.dataframe[self.cohortVariable].dropna()))):

            template[f"Cohort-{self.cohortVariable}"].append(cohort)

            cohortDataframe = reduceDataframe(self.dataframe,
                                              self.classVariable,
                                              self.targetVariable,
                                              self.cohortVariable)

            cohortDataframe[self.targetVariable] = pd.get_dummies(cohortDataframe[self.targetVariable],
                                                                  drop_first = True)

            if cohort == "All": pass
            else: cohortDataframe = cohortDataframe[cohortDataframe[self.cohortVariable] == cohort]

            template["All"].append("{}/{} ({:.2f})".format(np.sum(cohortDataframe[self.targetVariable]),
                                                           len(cohortDataframe),
                                                           np.sum(cohortDataframe[self.targetVariable]) /\
                                                           len(cohortDataframe) * 100))

            for group in groups:

                groupDataframe = cohortDataframe[cohortDataframe[self.classVariable] == group]

                template[f"{self.classVariable} {group}"].append("{}/{} ({:.2f})".format(np.sum(groupDataframe[self.targetVariable]),
                                                                                         len(groupDataframe),
                                                                                         np.sum(groupDataframe[self.targetVariable]) /\
                                                                                         len(groupDataframe) * 100))

            cohortDataframe[self.targetVariable] = cohortDataframe[self.targetVariable].replace(0, "no").replace(1, "yes")

            result = PearsonChiSquareTest(dataframe = cohortDataframe,
                                          classVariable = self.classVariable,
                                          targetVariable = self.targetVariable).run()

            template["P value"].append("{:.3f}".format(result.lowerPvalue))

        table = pd.concat([table, pd.DataFrame(template)])

        return table