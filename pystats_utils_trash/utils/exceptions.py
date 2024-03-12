class NoPvalueError(Exception):


    def __init__(self, *args, msg = "no P-value calculated"):

        self.args = args
        self.msg = msg

        super().__init__(self.msg)


    def __str__(self) -> str:
        return " ".join([str(arg) for arg in self.args]) + self.msg