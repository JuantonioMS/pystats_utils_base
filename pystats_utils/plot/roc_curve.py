
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import plotnine as p9
import plotnine_prism as p9prism

from pystats_utils.plot import Plot
from pystats_utils.data_operations import reduceDataframe

class RocCurvePlot(Plot):


    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariable = targetVariable


    def cleanData(self):

        return reduceDataframe(self.dataframe,
                               self.classVariable,
                               self.targetVariable)



    def cookData(self, dataframe):
        return dataframe



    def runPlot(self, data, **kwargs):

        if not "title" in kwargs: kwargs["title"] = "ROC Curve Plot"
        if not "xlabel" in kwargs: kwargs["xlabel"] = "1 - Specificity (%)"
        if not "ylabel" in kwargs: kwargs["ylabel"] = "Sensitivity (%)"
        if not "palette" in kwargs: kwargs["palette"] = "pastel"

        classData = pd.get_dummies(data[self.classVariable],
                                   prefix = self.classVariable)

        plots = []
        for group in classData:

            fpr, tpr, thresholds = roc_curve(classData[group],
                                             data[self.targetVariable])

            info = pd.DataFrame({"FPR"       : fpr,
                                 "TPR"       : tpr,
                                 "Threshold" : thresholds,
                                 "opt" : tpr + (1 - fpr)})



            plot = (
                    p9.ggplot() +
                    p9.geom_step(info,
                                 p9.aes(x = "FPR",
                                        y = "TPR"),
                                 size = 1,
                                 colour = "black") +
                    p9.geom_point(pd.DataFrame({"FPR" : [info["FPR"].iloc[info["opt"].idxmax()]],
                                                "TPR" : [info["TPR"].iloc[info["opt"].idxmax()]]}),
                                  p9.aes(x = "FPR",
                                         y = "TPR"),
                                  size = 3,
                                  colour = "black",
                                  fill = "red") +
                    p9.geom_label(pd.DataFrame({"FPR" : [info["FPR"].iloc[info["opt"].idxmax()] - 0.15],
                                                "TPR" : [info["TPR"].iloc[info["opt"].idxmax()] + 0.05],
                                                "Label" : [f"Optimal >= {info['Threshold'].iloc[info['opt'].idxmax()]}"]}),
                                  p9.aes(x = "FPR",
                                         y = "TPR",
                                         label = "Label")) +
                    p9prism.theme_prism() +
                    p9.labs(title = kwargs['title']) +
                    p9.scale_y_continuous(name = kwargs['ylabel'],
                                      limits = (0,1))+
                    p9.scale_x_continuous(name = kwargs['xlabel'])
                    )

            plots.append(plot)

        return plots if len(plots) > 2 else plots[1]