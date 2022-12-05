import pandas as pd
import numpy as np
from pathlib import Path


testDataframe = pd.DataFrame({"group" : ["g1"] * 50 + ["g2"] * 50 + ["g3"] * 50,
                              "num1"  : np.concatenate((np.random.randn(50) * 3 + (np.random.rand() * 10),
                                                        np.random.randn(50) * 3 + (np.random.rand() * 10),
                                                        np.random.randn(50) * 3 + (np.random.rand() * 10))),
                              "num2"  : np.concatenate((np.random.randn(50) * 2 + (np.random.rand() * 10),
                                                        np.random.randn(50) * 4 + (np.random.rand() * 10),
                                                        np.random.randn(50) * 6 + (np.random.rand() * 10))),
                               "num3"  : np.concatenate((np.random.chisquare(2, 50) * (np.random.rand() * 10) + (np.random.rand() * 10),
                                                        np.random.chisquare(2, 50) * (np.random.rand() * 10) + (np.random.rand() * 10),
                                                        np.random.chisquare(2, 50) * (np.random.rand() * 10) + (np.random.rand() * 10))),

                               "num4"  : np.concatenate((np.random.randn(100) * 3 + (np.random.rand() * 10),
                                                         np.random.randn(50) * 3 + (np.random.rand() * 10))),
                               "num5"  : np.concatenate((np.random.randn(100) * 2 + (np.random.rand() * 10),
                                                         np.random.randn(50) * 6 + (np.random.rand() * 10))),
                               "num6"  : np.concatenate((np.random.chisquare(2, 100) * (np.random.rand() * 10) + (np.random.rand() * 10),
                                                         np.random.chisquare(2, 50) * (np.random.rand() * 10) + (np.random.rand() * 10))),

                               "num7"  : np.random.randn(150) * (np.random.rand() * 10) + (np.random.rand() * 10),
                               "num8"  : np.concatenate((np.random.randn(50) * 0.5 + 2,
                                                         np.random.randn(50) * 1.0 + 2,
                                                         np.random.randn(50) * 1.5 + 2)),
                               "num9"  : np.random.chisquare(2, 150) * (np.random.rand() * 10) + (np.random.rand() * 10),

                               "cat1"  : ["a" if i < 0.2 else "b" for i in np.random.rand(50)] +\
                                         ["a" if i < 0.5 else "b" for i in np.random.rand(50)] +\
                                         ["a" if i < 0.8 else "b" for i in np.random.rand(50)],
                               "cat2"  : ["a" if i <= 0.33 else "b" if 0.33 < i < 0.66 else "c" for i in np.random.rand(50)] +\
                                         ["a" if i <= 0.2 else "b" if 0.2 < i < 0.8 else "c" for i in np.random.rand(50)] +\
                                         ["a" if i <= 0.1 else "b" if 0.1 < i < 0.3 else "c" for i in np.random.rand(50)],

                               "cat3"  : ["a" if i < 0.2 else "b" for i in np.random.rand(100)] +\
                                         ["a" if i < 0.8 else "b" for i in np.random.rand(50)],
                               "cat4"  : ["a" if i <= 0.33 else "b" if 0.33 < i < 0.66 else "c" for i in np.random.rand(100)] +\
                                         ["a" if i <= 0.1 else "b" if 0.1 < i < 0.3 else "c" for i in np.random.rand(50)],

                               "cat5"  : ["a" if i < 0.5 else "b" for i in np.random.rand(150)],
                               "cat6"  : ["a" if i <= 0.33 else "b" if 0.33 < i < 0.66 else "c" for i in np.random.rand(150)]})

testDataframe.to_csv(f"{Path(__file__).parent}/../database/database_test_r.csv", index = False)


def similarity(a, b, threshold = 0.99999, decimal = 1e-50):

    #  Identidad
    if a == b: result = True

    #  Uno es cero y otro casi cero
    elif a == 0.0 or b == 0.0:

        aux = a if a != 0 else b

        result =  aux <= decimal

    #  Ambos son casi cero
    elif a <= decimal and b <= decimal: result = True

    #  Otro cálculo
    else: result = (1 - abs(a - b) / (a + b)) >= threshold


    return result