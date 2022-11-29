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

                    if len(set(workDataframe[column])) == 1:
                        continue

                    template["Variable_type"] = ["categorical"]

                    testResult = PearsonChiSquareTest(dataframe = workDataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()

                    if len(set(workDataframe[column])) == 2:

                        tag = list(set(workDataframe[column]))
                        tag.sort()
                        tag = tag[1]

                        template["Variable"] = [f"{column}_{tag}"]

                    #  Es dicotómica
                    if len(set(workDataframe[column])) == 2:

                        auxDataseries = pd.get_dummies(workDataframe[column],
                                                       drop_first = True)

                        auxDataseries = auxDataseries[auxDataseries.columns[0]]

                        allAbsolute = np.sum(auxDataseries)
                        allRelative = round(allAbsolute / len(auxDataseries) * 100, 3)

                        template["All"] = [f"{allAbsolute} ({allRelative})"]

                        for group in groups:

                            aux = workDataframe[workDataframe[self.classVariable] == group]

                            if len(set(aux[column])) == 1:

                                auxDataseries = pd.get_dummies(aux[column],
                                                               drop_first = False)

                                if auxDataseries.columns[0] == tag:
                                    template[group] = [f"{len(auxDataseries)} (100.0)"]
                                else:
                                    template[group] = [f"0 (0.0)"]


                            else:
                                auxDataseries = pd.get_dummies(aux[column],
                                                            drop_first = True)

                                auxDataseries = auxDataseries[auxDataseries.columns[0]]

                                groupAbsolute = np.sum(auxDataseries)
                                groupRelative = round(groupAbsolute / len(auxDataseries) * 100, 3)

                                template[group] = [f"{groupAbsolute} ({groupRelative})"]

                    #  No es dicotómica
                    else:

                        auxDataframe = pd.get_dummies(workDataframe[column]).replace(1, "yes").replace(0, "no")
                        auxDataframe = auxDataframe.join(workDataframe[self.classVariable])

                        for column in [column for column in auxDataframe.columns if column != self.classVariable]:

                            template["Variable"].append(f"----> {column}")
                            template["Variable_type"].append("categorical")
                            template["Normality"].append("")
                            template["Homocedasticity"].append("")

                            testResult = PearsonChiSquareTest(dataframe = auxDataframe,
                                                              classVariable = self.classVariable,
                                                              targetVariable = column).run()

                            template["P_value"].append(round(testResult.pvalue, 3))
                            template["Test"].append(testResult.test)

                            auxWorkDataframe = reduceDataframe(auxDataframe,
                                                               self.classVariable, column)

                            if len(set(auxWorkDataframe[column])) == 1:

                                if "yes" in set(auxWorkDataframe[column]):
                                    template["All"].append(f"{len(auxWorkDataframe)} (100.0)")
                                else:
                                    template["All"].append(f"0 (100.0)")

                                for group in groups:

                                    groupAuxWorkDataframe = auxWorkDataframe[auxWorkDataframe[self.classVariable] == group]

                                    if auxWorkDataframe.columns[0] == "yes":
                                        template[group].append(f"{len(groupAuxWorkDataframe)} (100.0)")
                                    else:
                                        template[group].append(f"0 (100.0)")

                            else:

                                numAuxWorkDataframe = auxWorkDataframe.replace("yes", 1).replace("no", 0)

                                allAbsolute = np.sum(numAuxWorkDataframe[column])

                                allRelative = round(allAbsolute / len(numAuxWorkDataframe) * 100, 3)

                                template["All"].append(f"{allAbsolute} ({allRelative})")

                                for group in groups:

                                    groupAuxWorkDataframe = auxWorkDataframe[auxWorkDataframe[self.classVariable] == group]

                                    numGroupAuxWorkDataframe = groupAuxWorkDataframe.replace("yes", 1).replace("no", 0)

                                    groupAbsolute = np.sum(numGroupAuxWorkDataframe[column])
                                    groupRelative = round(groupAbsolute / len(numGroupAuxWorkDataframe) * 100, 3)

                                    template[group].append(f"{groupAbsolute} ({groupRelative})")


                #  Sección común
                template["P_value"] = ["{:.3f}".format(testResult.lowerPvalue)]
                template["Test"] = [testResult.test]

                table = pd.concat([table,
                                   pd.DataFrame(template)])

        return table