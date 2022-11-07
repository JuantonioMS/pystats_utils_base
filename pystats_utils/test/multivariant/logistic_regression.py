import statsmodels.formula.api as smf

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.utils import resample


from pystats_utils.test.multivariant import Multivariant

class LogisticRegression(Multivariant):

    def runTest(self,
                workingData: pd.DataFrame,
                formula: str) -> dict:

        classVariable = formula.split("~")[0].strip(" ")

        #  Model
        model = smf.logit(formula, data = workingData).fit(disp = 0)

        #  Result

        resCoef = model.params
        resOr = np.exp(resCoef)

        resPvalue = model.pvalues

        resErr = model.bse
        resConf = model.conf_int(0.05)
        resConfOr = np.exp(resConf)

        auxDataframe = workingData.copy()
        auxDataframe.drop(classVariable,
                          inplace = True,
                          axis = 1)

        result = pd.concat([resOr, resConfOr, resPvalue,
                            resCoef, resConf,
                            resErr], axis = 1)

        result = result.reset_index()

        result = result.set_axis(["Predictor",
                                  "aOR", "CI 2.5%", "CI 97.5%", "P value",
                                  "Coef", "Raw CI 2.5%", "Raw CI 97.5%",
                                  "Standard Err."], axis = 1)



        #  ROC curve info
        prediction = model.predict()

        # Bootstrapping

        aucROCs = []
        cutOffs = []
        accuracies = []
        errors = []
        precisions = []
        recalls = []
        f1s = []

        for _ in range(2000):

                bootExpectedClass, bootPrediciton = resample(workingData[classVariable],
                                                             prediction)

                aucROC = roc_auc_score(bootExpectedClass,
                                       bootPrediciton)

                fpr, tpr, thresholds = roc_curve(bootExpectedClass,
                                                 bootPrediciton)

                aux = np.arange(len(tpr))
                roc = pd.DataFrame({'tf' : pd.Series(abs(tpr - (1-fpr)), index = aux),
                                    'thresholds' : pd.Series(thresholds, index = aux)})

                threshold = list(roc.iloc[(roc.tf).abs().argsort()[:1]]["thresholds"])[0]

                #  Confusion matrix info

                expectedClass = bootExpectedClass
                predictedClass = [1 if pred > threshold else 0 for pred in prediction]

                accuracy = accuracy_score(expectedClass, predictedClass)
                error = 1 - accuracy
                precision = precision_score(expectedClass, predictedClass)
                recall = recall_score(expectedClass, predictedClass)
                f1 = f1_score(expectedClass, predictedClass)

                aucROCs.append(aucROC)
                cutOffs.append(threshold)
                accuracies.append(accuracy)
                errors.append(error)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)


        aucROC = roc_auc_score(workingData[classVariable],
                               prediction)

        fpr, tpr, thresholds = roc_curve(workingData[classVariable],
                                         prediction)

        aux = np.arange(len(tpr))
        roc = pd.DataFrame({'tf' : pd.Series(abs(tpr - (1-fpr)), index = aux),
                            'thresholds' : pd.Series(thresholds, index = aux)})

        threshold = list(roc.iloc[(roc.tf).abs().argsort()[:1]]["thresholds"])[0]

        #  Confusion matrix info

        expectedClass = workingData[classVariable]
        predictedClass = [1 if pred > threshold else 0 for pred in prediction]

        accuracy = accuracy_score(expectedClass, predictedClass)
        error = 1 - accuracy
        precision = precision_score(expectedClass, predictedClass)
        recall = recall_score(expectedClass, predictedClass)
        f1 = f1_score(expectedClass, predictedClass)

        values = [aucROC, threshold, accuracy, error, precision, recall, f1]
        boots = [aucROCs, thresholds, accuracies, errors, precisions, recalls, f1s]




        lowerCI = [original - 1.97 * np.std(boot)   for original, boot in zip(values, boots)]
        lowerCI = [ci if ci > 0 else 0.0 for ci in lowerCI]


        upperCI = [original + 1.97 * np.std(boot)   for original, boot in zip(values, boots)]
        upperCI = [ci if ci < 1 else 1.0 for ci in upperCI]


        summary = pd.DataFrame({"Value"    : values,
                                "CI 2.5%"  : lowerCI,
                                "CI 97.5%" : upperCI},
                               index = ["Area under curve ROC",
                                        "Best Cutoff",
                                        "Accuracy",
                                        "Error",
                                        "Precision",
                                        "Recall",
                                        "F1"])

        prediction = pd.Series(prediction,
                               index = workingData.index)

        return {"params" : result,
                "model" : model,
                "summary" : summary,
                "prediction" : prediction}