import numpy as np
import pandas as pd

from pystats_utils.statistical_test.bivariate.normality.kolmogorov_test import KolmogorovSmirnovTest
from pystats_utils.statistical_test.bivariate.homocedasticity.levene_test import LeveneTest

from pystats_utils.statistical_test.bivariate.value_comparison.t_test import StudentTTest
from pystats_utils.statistical_test.bivariate.value_comparison.welch_test import WelchTest
from pystats_utils.statistical_test.bivariate.value_comparison.mann_whitney_u_test import MannWhitneyUTest
from pystats_utils.statistical_test.bivariate.factor_comparison.pearson_chi_square_test import PearsonChiSquareTest

from pystats_utils.configuration.key_words import P_VALUE

class BivariantTable:



    def __init__(self,
                 database = None,
                 dependentVariable: str = "",
                 excludedVariables: list = []):


        self.__database = database
        self.__dependentVariable = dependentVariable
        self.__excludedVariables = excludedVariables



    def run(self) -> pd.DataFrame:

        #  Variable, variable_type
        #  all, groups, ...
        #  p_value, test, normality, homocedasticity


        database = self.__database.drop(self.__dependentVariable,
                                        "variable is None")

        groups = self.__getGroups()

        dataframe = []
        for variable in database.iterConfiguration():

            if not variable.type in ["integer", "float", "percentage", "ranking",
                                     "nominal", "ordinal", "binomial", "boolean",
                                     "multilabel"]: continue

            if variable.name in self.__excludedVariables + [self.__dependentVariable]: continue

            registers = self.__cleanDatabase(variable.name)

            if not registers: continue

            #  Numerical section
            if variable.type in ["integer", "float", "percentage", "ranking"]:

                dataframe.append(self.__runNumerical(groups = groups,
                                                     registers = registers,
                                                     variable = variable))

            #  Nominal section
            elif variable.type in ["nominal", "ordinal",
                                   "binomial", "boolean",
                                   "multilabel"]:

                dataframe += self.__runCategorical(groups = groups,
                                                   registers = registers,
                                                   variable = variable)

        groups = [f"{group} (n={len([register for register in database.iterRegisters() if register[self.__dependentVariable] == group])})" for group in groups]

        dataframe = pd.DataFrame(dataframe,
                                 columns = ["variable", "type",
                                            f"all (n={len(database._registers)})"] + \
                                           groups + \
                                           [P_VALUE, "test", "normality", "homocedasticity"])

        return dataframe



    #%%  PREPROCESSING METHODS__________________________________________________________________________________________



    def __getGroups(self) -> list:

        return sorted(list({register[self.__dependentVariable] \
                            for register in self.__database.iterRegisters() \
                            if not register[self.__dependentVariable] is None}))



    def __cleanDatabase(self, variable: str) -> list:

        return [register \
                for register in self.__database.iterRegisters() \
                if not register[variable] is None]



    #%%  NUMERICAL SECTION::____________________________________________________________________________________________



    def __runNumerical(self,
                       groups: list = [],
                       registers: list = [],
                       variable: str = "") -> list:

        return [variable.name, variable.type] + \
               self.__runNumericalDescriptive(groups = groups,
                                              registers = registers,
                                              variable= variable) + \
               self.__runNumericalTest(variable = variable)



    def __runNumericalDescriptive(self,
                                  groups: list = [],
                                  registers: list = [],
                                  variable: str = "") -> list:

        descriptive = []
        for group in ["all"] + groups:

            if group == "all": filter = groups
            else: filter = [group]

            values = [register[variable.name] \
                      for register in registers \
                      if register[self.__dependentVariable] in filter]

            if not values:
                descriptive.append("No data")

            else:
                descriptive.append("{:.3f} ({:.3f} - {:.3f})".format(np.percentile(values, 50),
                                                                    np.percentile(values, 25),
                                                                    np.percentile(values, 75)))

        return descriptive



    def __runNumericalTest(self,
                           variable: str = ""):

        #  Normality
        normalityResults = KolmogorovSmirnovTest(database = self.__database,
                                                 independentVariable = variable.name,
                                                 dependentVariable = self.__dependentVariable).run()

        #  Non normality and unknown variance
        if min(normalityResults[element][P_VALUE] for element in normalityResults) < 0.05:
            info = [False, ""]

        else:

            homocedasticityResult = LeveneTest(database = self.__database,
                                               independentVariable = variable.name,
                                               dependentVariable = self.__dependentVariable).run()

            #  Normality and non homocedasticity
            if min(normalityResults[element][P_VALUE] for element in homocedasticityResult) < 0.05: info = [True, False]

            #  Normality and Homocedasticity
            else: info = [True, True]

        if info[0] == True and info[1] == True: Test = StudentTTest #  Normality and Homocedasticity
        elif info[0] == True and info[1] == False: Test = WelchTest #  Normality and non homocedasticity
        elif info[0] == False: Test = MannWhitneyUTest #  No normality and unknown variance

        test = Test(database = self.__database,
                    independentVariable = variable.name,
                    dependentVariable = self.__dependentVariable)

        testResult = test.run()

        pValue = min(testResult[element][P_VALUE] for element in testResult)

        return [round(pValue, 3), test.prettyName] + info



    #%%  CATEGORICAL SECTION::__________________________________________________________________________________________



    def __runCategorical(self,
                         groups: list = [],
                         registers: list = [],
                         variable: str = "") -> list:

        descriptive = self.__runCategoricalDescriptive(groups = groups,
                                                       registers = registers,
                                                       variable = variable)

        test = self.__runCategoricalTest(variable = variable)

        results = []

        for key in descriptive:

            try:
                results.append(descriptive[key] + [round(test[key][P_VALUE], 3)] + ["Pearson Chi Square Test", "", ""])

            except KeyError:
                results.append(descriptive[key] + ["", "Pearson Chi Square Test", "", ""])

        return results



    def __runCategoricalDescriptive(self,
                                    groups: list = [],
                                    registers: list = [],
                                    variable: str = "") -> dict:

        dataframe = variable.variableToDataframe(registers)

        descriptive = {"all": [variable.name, variable.type] + ([""] * (len(groups) + 1))}
        for column in dataframe:

            row = [f"--> {column}", "boolean"]
            for group in ["all"] + groups:

                if group == "all": filter = groups
                else: filter = [group]

                registerIndexes = [register.id for register in registers if register[self.__dependentVariable] in filter]
                auxDataframe = dataframe[dataframe.index.isin(registerIndexes)]

                if auxDataframe.empty:
                    row.append("0 (0.0)")
                else:

                    row.append("{} ({:.2f})".format(sum(auxDataframe[column]),
                                                    sum(auxDataframe[column]) / len(auxDataframe[column]) * 100))

            descriptive[column] = row

        return descriptive



    def __runCategoricalTest(self,
                             variable: str = "") -> dict:

        testResult = PearsonChiSquareTest(database = self.__database,
                                          independentVariable = variable.name,
                                          dependentVariable = self.__dependentVariable).run()

        return testResult