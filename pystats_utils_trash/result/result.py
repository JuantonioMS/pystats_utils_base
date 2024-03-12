class Result:

    alpha = 0.05

    def __init__(self, **kwargs):

        for key, value in kwargs.items():

            setattr(self, key, value)

        if hasattr(self, "pvalue"):

            self.significance = self.lowerPvalue < self.alpha


    def __str__(self) -> str:

        msg = ["Result"]

        for key, value in self.__dict__.items():

            msg.append(f"\t{key}: {value}")

        return "\n".join(msg)


    def __repr__(self) -> str:
        return str(self)


    @property
    def lowerPvalue(self):

        if isinstance(self.pvalue, float): return self.pvalue

        else:

            #  Diccionario anidado
            if any([isinstance(aux, dict) for aux in self.pvalue.values()]):
                return min([aux2 for aux in self.pvalue.values() for aux2 in aux.values()])

            #  Diccionario no anidado
            else:
                return min([aux for aux in self.pvalue.values()])