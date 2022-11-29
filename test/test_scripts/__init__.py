import pandas as pd
from pathlib import Path

testDataframe = pd.read_csv(Path(Path(__file__).parent, "../database/example_python.csv"), sep = ",")

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