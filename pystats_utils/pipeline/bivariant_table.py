import pandas as pd
import numpy as np

from pystats_utils.test.normality import KolmogorovSmirnovTest
from pystats_utils.test.homocedasticity import LeveneTest, BrownForsythTest
from pystats_utils.test.value_comparison import StudentTTest, WelchTest, MannWhitneyUTest
from pystats_utils.test.categorical_comparison import PearsonChiSquareTest

from pystats_utils.data_operations import isCategorical
from pystats_utils.data_operations import reduceDataframe


class BivariantTable:

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 excludedVariables: list = []):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.excludedVariables = excludedVariables


    def run(self) -> pd.DataFrame:

        startColumns = {"Variable" : pd.Series(dtype = "str"),
                        "All" : pd.Series(dtype = str)}

        groups = list(set(list(self.dataframe[self.classVariable].dropna())))

        groups.sort()

        groupColumns = dict([(group, pd.Series(dtype = "str")) for group in groups])

        endColumns = {"P_value" : pd.Series(dtype = "float"),
                      "Test" : pd.Series(dtype = "str"),
                      "Variable_type" : pd.Series(dtype = "str"),
                      "Normality" : pd.Series(dtype = "str"),
                      "Homocedasticity" : pd.Series(dtype = "str")}

        header = {}
        header.update(startColumns)
        header.update(groupColumns)
        header.update(endColumns)

        table = pd.DataFrame(header)

        for column in self.dataframe:

            if column not in self.excludedVariables + [self.classVariable]:

                workDataframe = reduceDataframe(self.dataframe,
                                                self.classVariable,
                                                column)

                template = dict([(key, [""]) for key in header])
                template["Variable"] = [column]


                #  Numerical section
                if  not isCategorical(self.dataframe, column):

                #  Categorical section

                    template["Variable_type"] = ["numerical"]


                    #  Testear la normalidad
                    normalityResult = KolmogorovSmirnovTest(dataframe = workDataframe,
                                                            classVariable = self.classVariable,
                                                            targetVariable = column).run()

                    template["Normality"] = ["Yes" if not normalityResult.significance else "No"]


                    #  Testar la homocedasticidad
                    if not normalityResult.significance:

                        homocedasticityResult = LeveneTest(dataframe = workDataframe,
                                                           classVariable = self.classVariable,
                                                           targetVariable = column).run()

                    else:

                        homocedasticityResult = BrownForsythTest(dataframe = workDataframe,
                                                                 classVariable = self.classVariable,
                                                                 targetVariable = column).run()

                    template["Homocedasticity"] = ["Yes" if not homocedasticityResult.significance else "No"]


                    #  Testear la comparacion
                    if not normalityResult.significance: #  Si es paramétrico

                        if not homocedasticityResult.significance: #  Si las varianzas son iguales

                            testResult = StudentTTest(dataframe = workDataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()

                        else: #  Si las varianzas no son iguales

                            testResult = WelchTest(dataframe = workDataframe,
                                                   classVariable = self.classVariable,
                                                   targetVariable = column).run()

                    else: #  Si no es paramétrico

                        testResult = MannWhitneyUTest(dataframe = workDataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()

                    template["All"] = ["{:.2f} ({:.2f} - {:.2f})".format(np.mean(workDataframe[column]),
                                                                        np.percentile(workDataframe[column], 25),
                                                                        np.percentile(workDataframe[column], 75))]

                    for group in groups:

                        aux = workDataframe[workDataframe[self.classVariable] == group]

                        template[group] = ["{:.2f} ({:.2f} - {:.2f})".format(np.mean(aux[column]),
                                                                            np.percentile(aux[column], 25),
                                                                            np.percentile(aux[column], 75))]

                        template["P_value"] = ["{:.3f}".format(testResult.lowerPvalue)]
                        template["Test"] = [testResult.test]

                #  Categorical section
                else:

                    template["Variable_type"] = ["categorical"]

                    testResult = PearsonChiSquareTest(dataframe = workDataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()

                    template["P_value"] = ["{:.3f}".format(testResult.lowerPvalue)]
                    template["Test"] = [testResult.test]

                    for tag in sorted(set(workDataframe[column])):

                        auxDataframe = pd.get_dummies(workDataframe[column])
                        auxDataframe = auxDataframe.join(workDataframe[self.classVariable])

                        template["Variable"].append(f"----> {column}-{tag}")
                        template["Variable_type"].append("categorical")
                        template["Normality"].append("")
                        template["Homocedasticity"].append("")

                        try:
                            template["All"].append("{} ({:.2f})".format(np.sum(auxDataframe[tag]),
                                                                        np.sum(auxDataframe[tag]) /\
                                                                        len(auxDataframe[tag])))

                        except ZeroDivisionError: template["All"].append("{} ({:.2f})".format(0, 0))

                        for group in groups:

                            try:
                                template[group].append("{} ({:.2f})".format(np.sum(auxDataframe[auxDataframe[self.classVariable] == group][tag]),
                                                                            np.sum(auxDataframe[auxDataframe[self.classVariable] == group][tag]) /\
                                                                            len(auxDataframe[auxDataframe[self.classVariable] == group][tag])))

                            except ZeroDivisionError: template[group].append("{} ({:.2f})".format(0, 0))

                        auxDataframe = auxDataframe.replace(1, "yes").replace(0, "no")

                        testResult = PearsonChiSquareTest(dataframe = auxDataframe,
                                                          classVariable = self.classVariable,
                                                          targetVariable = tag).run()

                        template["P_value"].append("{:.3f}".format(testResult.lowerPvalue))
                        template["Test"].append(testResult.test)

                table = pd.concat([table,
                                   pd.DataFrame(template)])

        return table