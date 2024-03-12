class Table:

    def run(self, **kwargs):

        dataframe = self.cleanData()

        data = self.cookData(dataframe)

        result = self.runTable(data, **kwargs)

        return result

from pystats_utils.table.risk_table import RiskTable
