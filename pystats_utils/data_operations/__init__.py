import pandas as pd

def reduceDataframe(dataframe: pd.DataFrame,
                    *columns) -> pd.DataFrame:

    """Selecciona solo las columnas indicadas y elimina todos las filas con algún elemento vacío"""

    auxdf = dataframe[list(columns)].dropna()

    return auxdf


def isCategorical(dataframe: pd.DataFrame,
                  column) -> bool:

    aux = reduceDataframe(dataframe, column)

    return type(list(aux[column])[0]) == str