class Plot:

    def run(self, **kwargs):

        dataframe = self.cleanData()

        data = self.cookData(dataframe)

        result = self.runPlot(data, **kwargs)

        return result

from pystats_utils.plot.ridge_plot import RidgePlot
from pystats_utils.plot.kaplan_meier_curve import KaplanMeierCurvePlot
from pystats_utils.plot.roc_curve import RocCurvePlot