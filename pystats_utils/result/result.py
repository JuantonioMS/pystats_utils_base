class Result:


    def __init__(self, **kwargs):

        for key, value in kwargs.items():

            setattr(self, key, value)


    def __str__(self) -> str:

        msg = ["Result"]

        for key, value in self.__dict__.items():

            msg.append(f"\t{key}: {value}")

        return "\n".join(msg)


    def __repr__(self) -> str:
        return str(self)