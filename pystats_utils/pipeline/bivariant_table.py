import pandas as pd

from pystats_utils.test.normality import KolmogorovSmirnovTest
from pystats_utils.test.homocedasticity import LeveneTest, BrownForsythTest
from pystats_utils.test.value_comparison import StudentTTest, WelchTest, MannWhitneyUTest
from pystats_utils.test.categorical_comparison import PearsonChiSquareTest

from pystats_utils.data_operations import isCategorical


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

                template = dict([(key, [""]) for key in header])

                print(column)
                template["Variable"] = [column]

                #  Categorical section
                if isCategorical(self.dataframe, column):

                    testResult = PearsonChiSquareTest(dataframe = self.dataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()

                    template["Variable_type"] = ["categorical"]

                #  Numerical section
                else:

                    #  Testear la normalidad
                    normalityResult = KolmogorovSmirnovTest(dataframe = self.dataframe,
                                                            classVariable = self.classVariable,
                                                            targetVariable = column).run()

                    normality = "Yes" if not normalityResult.significance else "No"

                    #  Testar la homocedasticidad
                    if not normalityResult.significance:

                        homocedasticityResult = LeveneTest(dataframe = self.dataframe,
                                                           classVariable = self.classVariable,
                                                           targetVariable = column).run()

                    else:

                        homocedasticityResult = BrownForsythTest(dataframe = self.dataframe,
                                                                 classVariable = self.classVariable,
                                                                 targetVariable = column).run()

                    homocedasticity = "Yes" if not homocedasticityResult.significance else "No"

                    #  Testear la comparacion

                    if not normalityResult.significance: #  Si es paramétrico

                        if not normalityResult.significance: #  Si las varianzas son iguales

                            testResult = StudentTTest(dataframe = self.dataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()

                        else: #  Si las varianzas no son iguales

                            testResult = WelchTest(dataframe = self.dataframe,
                                                   classVariable = self.classVariable,
                                                   targetVariable = column).run()

                    else: #  Si no es paramétrico

                        testResult = MannWhitneyUTest(dataframe = self.dataframe,
                                                      classVariable = self.classVariable,
                                                      targetVariable = column).run()




                    template["Variable_type"] = ["numerical"]
                    template["Normality"] = [normality]
                    template["Homocedasticity"] = [homocedasticity]

                template["P_value"] = [testResult.pvalue]
                template["Test"] = [testResult.test]

                table = pd.concat([table,
                                   pd.DataFrame(template)])

        return table