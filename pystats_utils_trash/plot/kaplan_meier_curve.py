
import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter

import plotnine as p9
import plotnine_prism as p9prism

from pystats_utils.plot import Plot
from pystats_utils.data_operations import reduceDataframe

class KaplanMeierCurvePlot(Plot):

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 eventVariable: str = "",
                 timeVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.eventVariable = eventVariable

        self.timeVariable = timeVariable



    def cleanData(self):

        return reduceDataframe(self.dataframe,
                               self.classVariable,
                               self.timeVariable,
                               self.eventVariable)



    def cookData(self, dataframe):

        dataframe[self.eventVariable] = pd.get_dummies(dataframe[self.eventVariable],
                                                       drop_first = True)

        return dataframe



    def runPlot(self, data, **kwargs):

        if not "title" in kwargs: kwargs["title"] = "Kaplan Meier Curve Plot"
        if not "xlabel" in kwargs: kwargs["xlabel"] = "Timeline"
        if not "ylabel" in kwargs: kwargs["ylabel"] = "Probability"
        if not "palette" in kwargs: kwargs["palette"] = "pastel"

        info = pd.DataFrame({"Timeline"   : pd.Series(dtype = "float"),
                             "Probabilty" : pd.Series(dtype = "float"),
                             "min"        : pd.Series(dtype = "float"),
                             "max"        : pd.Series(dtype = "float"),
                             "group"      : pd.Series(dtype = "str")})

        if self.classVariable:

            for group in sorted(data[self.classVariable].unique()):

                workData = data[data[self.classVariable] == group]

                model = KaplanMeierFitter()
                model.fit(workData[self.timeVariable],
                          workData[self.eventVariable])

                aux = pd.DataFrame({"Timeline"   : list(model.survival_function_["KM_estimate"].index),
                                    "Probabilty" : list(model.survival_function_["KM_estimate"]),
                                    "min"        : list(model.confidence_interval_["KM_estimate_lower_0.95"]),
                                    "max"        : list(model.confidence_interval_["KM_estimate_upper_0.95"]),
                                    "group"      : [f"{self.classVariable}_{group}"] * len(list(model.survival_function_["KM_estimate"].index))})

                info = pd.concat([info, aux],
                                 axis = 0)

        else:

            model = KaplanMeierFitter()
            model.fit(data[self.timeVariable],
                      data[self.eventVariable])

            aux = pd.DataFrame({"Timeline"   : list(model.survival_function_["KM_estimate"].index),
                                "Probabilty" : list(model.survival_function_["KM_estimate"]),
                                "min"        : list(model.confidence_interval_["KM_estimate_lower_0.95"]),
                                "max"        : list(model.confidence_interval_["KM_estimate_upper_0.95"]),
                                "group"      : ["Global"] * len(list(model.survival_function_["KM_estimate"].index))})

            info = pd.concat([info, aux],
                                axis = 0)

        plot = (
                p9.ggplot() +
                p9.geom_ribbon(info,
                               p9.aes(x = 'Timeline',
                                      ymin = 'min',
                                      ymax = 'max',
                                      fill = 'group'),
                               alpha = 0.3) +
                p9.geom_step(info,
                             p9.aes(x = 'Timeline',
                                    y = 'Probabilty',
                                    colour = 'group'),
                             size = 1) +
                p9prism.theme_prism() +
                p9.labs(title = kwargs['title'])+
                p9.scale_y_continuous(name = kwargs['ylabel'],
                                      limits = (0,1))+
                p9.scale_x_continuous(name = kwargs['xlabel'])
                )

        return plot