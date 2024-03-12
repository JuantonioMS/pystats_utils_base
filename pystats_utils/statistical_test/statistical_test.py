class StatisticalTest:



    def __init__(self,
                 database,
                 independentVariable: str):

        self._database = database
        self._indpendentVariable = independentVariable



    @property
    def name(self) -> str:
        return self.__class__.__name__



    @property
    def prettyName(self):
        return "".join([f" {char}" if char.isupper() else char for char in self.name]).strip(" ")



    def run(self):

        """
        Step 1. Clean data
        Step 2. Pre-process data
        Step 3. Run test
        Step 4. Format results
        """

        variables = self._mergeVariables()

        registers = self._cleanData(*variables)

        data = self._preProcessData(registers)

        result = self._runTest(data)

        return self._formatResults(result)



    def _mergeVariables(self):
        return [self._indpendentVariable]



    def _cleanData(self, *args) -> list:

        auxRegisters = [register for register in self._database.iterRegisters()]

        for variable in args: auxRegisters = [register for register in auxRegisters if not register[variable] is None]

        return auxRegisters



    def _preProcessData(self, registers):
        return [register[self._indpendentVariable] for register in registers]



    def _runTest(self, data) -> dict:
        return {}



    def _formatResults(self, kwargs):
        return kwargs