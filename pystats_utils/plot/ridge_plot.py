import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pystats_utils.plot import Plot
from pystats_utils.data_operations import reduceDataframe

class RidgePlot(Plot):
    
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
        
        if not "title" in kwargs:
            kwargs["title"] = "Title"
        
        sns.set_theme(style = "white",
                      rc = {"axes.facecolor" : (0, 0, 0, 0)})
        
        palette = sns.color_palette("Set2", 12)
        
        g = sns.FacetGrid(data,
                          palette = palette,
                          row = self.classVariable,
                          hue = self.classVariable,
                          aspect = 9,
                          height = 1.2)
        
        g.map_dataframe(sns.kdeplot,
                        x = self.targetVariable,
                        fill = True,
                        alpha = 1)
        
        g.map_dataframe(sns.kdeplot,
                        x = self.targetVariable,
                        color = "black")
        
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, 0.2,
                    label, color = "black",
                    fontsize = 13,
                    ha = "left", va = "center",
                    transform = ax.transAxes)
            
        g.map(label, self.classVariable)
        
        g.fig.subplots_adjust(hspace = -0.5)
        
        g.set_titles("")
        g.set(yticks = [], xlabel = self.targetVariable)
        g.despine(left = True)
        
        plt.suptitle(kwargs["title"], y = 0.98)
        
        return g
        
        
    def run(self, **kwargs):
        
        dataframe = self.cleanData()
        
        data = self.cookData(dataframe)
        
        result = self.runPlot(data, **kwargs)
        
        return result