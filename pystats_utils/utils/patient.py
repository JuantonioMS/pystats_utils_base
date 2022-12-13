from datetime import datetime as dt
from datetime import timedelta as td

class Antibiotic:


    def __init__(self, info):

        self.name = info[0].capitalize()
        self.activity = info[2].upper() if info[2] != "nan" else ""
        self.start = info[4]
        self.end = info[5]

        self.reformatDate()
        self.reformatActivity()


    def reformatActivity(self):

        if self.activity:

            if self.activity in ["I", "S", "NR", "SUSC", "SUSCEPTIBLE", "SUSCPETIBLE"]:
                self.activity = "S"

            elif self.activity == "R":
                pass

            else:
                print(self.activity)
                raise ValueError

        else: self.activity = "S"


    def reformatDate(self):

        if "-" in self.start:
            self.start = dt.strptime(self.start, "%Y-%m-%d")
        elif "/" in self.start:
            self.start = dt.strptime(self.start, "%d/%m/%Y")

        if "-" in self.end:
            self.end = dt.strptime(self.end, "%Y-%m-%d")
        elif "/" in self.end:
            self.end = dt.strptime(self.end, "%d/%m/%Y")

        self.duration = (self.end - self.start).days + 1


    def __hash__(self): return hash(f"{self.name}_{self.activity}")

    def __str__(self): return f"{self.name}: {self.activity}\n\t{self.start} -> {self.end}"

    def __repr__(self) -> str: return str(self)

    def __eq__(self, other): return f"{self.name}_{self.activity}" == f"{other.name}_{other.activity}"

class Patient:


    def __init__(self, info):

        self.extractInfo(info)
        self.bloodculture = dt.strptime(self.bloodculture, "%m/%d/%y")


    def extractInfo(self, info):

        for column in info.index:
            value = info[column]

            if "ATB" not in column: setattr(self, column.lower(), value.lower() if isinstance(value, str) else value)

        columns = info.index[16:]

        self.treatment = []
        for index, column in enumerate(columns[::7]):

            antibioticInfo = [f"{info[aux]}" for aux in columns[index * 7 : index * 7 + 7]]

            if antibioticInfo[0] != "nan":
                self.treatment.append(Antibiotic(antibioticInfo))

    # Funcionalidades básicos

    @property
    def dates(self):

        dates = []

        for atb in self.treatment:
            dates.append(atb.start)
            dates.append(atb.end)

        dates.sort()

        return dates

    @property
    def firstDay(self): return self.dates[0]

    @property
    def lastDay(self): return self.dates[-1]

    @property
    def treatmentDuration(self): return (self.lastDay - self.firstDay).days + 1

    # Líneas de tratamiento

    @property
    def treatmentLine(self):

        timeline = [""] * self.treatmentDuration
        for dayIndex, day in enumerate((self.firstDay + td(n) for n in range(self.treatmentDuration))):

            for antibiotic in self.treatment:

                if antibiotic.start <= day <= antibiotic.end:

                    try: timeline[dayIndex].add(antibiotic)
                    except AttributeError: timeline[dayIndex] = {antibiotic}

        return timeline


    def translateTreatmentLine(self,
                               only: str = "all",
                               mergeInactive: bool = True,
                               condensed: bool = True,
                               days: int = 30,
                               start = False,
                               verbose: bool = False):

        timeline = []
        if only != "all":
            for day in self.treatmentLine:

                aux = set()
                for antibiotic in day:

                    if only == "active":
                        if antibiotic.activity == "S": aux.add(antibiotic)

                    elif only == "inactive":
                        if antibiotic.activity == "R": aux.add(antibiotic)

                timeline.append(aux)

        else:
            timeline = self.treatmentLine


        if start:

            if start >= self.firstDay:
                timeline = timeline[(start - self.firstDay).days:]

            else:
                a = [{}] * (self.firstDay - start).days
                timeline = a + timeline

        if days: timeline = timeline[:days]


        if condensed:
            auxTimeline = []

            for day in timeline:

                try:
                    if auxTimeline[-1] != day:

                        if verbose:
                            print(auxTimeline[-1])
                            print("___")
                            print(day)
                            print("----------------------------------------")

                        auxTimeline.append(day)

                except IndexError:
                    auxTimeline.append(day)

            timeline = auxTimeline

        translation = []
        for day in timeline:

            aux = set()
            for antibiotic in day:

                if only == "all":

                    if mergeInactive:
                        if antibiotic.activity == "R": aux.add("Inactive")
                        else: aux.add(antibiotic.name)

                    else:
                        aux.add(f"{antibiotic.name}_{antibiotic.activity}")

                else:
                    aux.add(antibiotic.name)

            translation.append(aux)


        for index in range(len(translation)):
            if not translation[index]: translation[index] = {"None"}


        return translation