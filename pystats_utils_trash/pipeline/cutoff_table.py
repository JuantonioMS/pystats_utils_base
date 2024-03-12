import pandas as pd
import numpy as np

from pystats_utils.data_operations import isCategorical
from pystats_utils.data_operations import reduceDataframe

from pystats_utils.test.categorical_comparison import ConfussionMatrixAnalysis

class CutoffTable:

    def __init__(self,
                 dataframe: pd.DataFrame = pd.DataFrame(),
                 classVariable: str = "",
                 targetVariable: str = ""):

        self.dataframe = dataframe

        self.classVariable = classVariable

        self.targetVariable = targetVariable



    def run(self):

        workDataframe = reduceDataframe(self.dataframe,
                                        self.classVariable,
                                        self.targetVariable)

        header = {"Cutoff"         : pd.Series(dtype = "str"),
                  "Proportion"     : pd.Series(dtype = "str"),
                  "True positive"  : pd.Series(dtype = "int"),
                  "False positive" : pd.Series(dtype = "int"),
                  "True negative"  : pd.Series(dtype = "int"),
                  "False negative" : pd.Series(dtype = "int"),
                  "Sensitivity"    : pd.Series(dtype = "str"),
                  "Specificity"    : pd.Series(dtype = "str"),
                  "PPV"            : pd.Series(dtype = "str"),
                  "NPV"            : pd.Series(dtype = "str"),
                  "Accuracy"       : pd.Series(dtype = "str")}

        table = pd.DataFrame(header)

        template = {col : [] for col in table}
        for num in range(max(workDataframe[self.targetVariable]) + 1):

            predicted = workDataframe[self.targetVariable] >= num
            predicted = predicted.replace(True, "yes").replace(False, "no")

            result = ConfussionMatrixAnalysis(dataframe = pd.DataFrame({"Expected"  : list(workDataframe[self.classVariable]),
                                                                        "Predicted" : list(predicted)}),
                                              classVariable = "Expected",
                                              targetVariable = "Predicted").run()

            template["Cutoff"].append(f"Var >= {num}")

            template["Proportion"].append("{:.2f}".format(np.sum(predicted.replace("yes", 1).replace("no", 0)) /\
                                                          len(predicted) * 100))

            template["True positive"].append(result.tp)
            template["False positive"].append(result.fp)
            template["True negative"].append(result.tn)
            template["False negative"].append(result.fn)

            template["Sensitivity"].append("{:.2f} ({:.2f} - {:.2f})".format(result.summary["Value"]["Sensitivity"],
                                                                             result.summary["CI 2.5%"]["Sensitivity"],
                                                                             result.summary["CI 97.5%"]["Sensitivity"]))

            template["Specificity"].append("{:.2f} ({:.2f} - {:.2f})".format(result.summary["Value"]["Specificity"],
                                                                             result.summary["CI 2.5%"]["Specificity"],
                                                                             result.summary["CI 97.5%"]["Specificity"]))

            template["PPV"].append("{:.2f} ({:.2f} - {:.2f})".format(result.summary["Value"]["PPV"],
                                                                             result.summary["CI 2.5%"]["PPV"],
                                                                             result.summary["CI 97.5%"]["PPV"]))

            template["NPV"].append("{:.2f} ({:.2f} - {:.2f})".format(result.summary["Value"]["NPV"],
                                                                             result.summary["CI 2.5%"]["NPV"],
                                                                             result.summary["CI 97.5%"]["NPV"]))

            template["Accuracy"].append("{:.2f} ({:.2f} - {:.2f})".format(result.summary["Value"]["Accuracy"],
                                                                             result.summary["CI 2.5%"]["Accuracy"],
                                                                             result.summary["CI 97.5%"]["Accuracy"]))

        table = pd.concat([table, pd.DataFrame(template)])

        return table


