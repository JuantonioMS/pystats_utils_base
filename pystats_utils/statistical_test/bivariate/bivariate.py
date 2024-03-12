from collections import defaultdict

from pystats_utils.statistical_test.statistical_test import StatisticalTest

from pystats_utils.configuration.key_words import P_VALUE, STATISTIC

class BivariateStatisticalTest(StatisticalTest):



    def __init__(self,
                 database,
                 independentVariable: str = "",
                 dependentVariable: str = ""):

        self._database = database
        self._independentVariable = independentVariable
        self._dependentVariable = dependentVariable



    def _mergeVariables(self):
        return [self._independentVariable, self._dependentVariable]



    def _preProcessData(self, registers) -> dict:

        auxData = defaultdict(list)

        for register in registers:
            auxData[register[self._dependentVariable]].append(register[self._independentVariable])

        return auxData



    def _runTest(self, data: dict) -> dict:

        results = {}

        for group1, values1 in data.items():
            for group2, values2 in data.items():

                if group1 != group2:

                    title = tuple(sorted([group1, group2]))

                    if title in results:
                        continue

                    statistic, pvalue =self._coreFunction(values1, values2)

                    results[title] = {P_VALUE : pvalue,
                                      STATISTIC : statistic}

        return results



    @staticmethod
    def _coreFunction(values1, values2):
        return None, None