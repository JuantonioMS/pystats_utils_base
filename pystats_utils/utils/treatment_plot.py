import pandas as pd
from plotnine import *
from plotnine_prism import *

def generatePlot(info, width = 0.9, height = 0.9, days = 30, title = "title"):

    mask = {"Caz-avi"               : "zCaz-avi", # CAZAVI   amarrillo
            "Inactive"              : "aInactive", # Inactivo gris
            "Amikacin"              : "bAmikacin", # Aminoglicosidos verde
            "Gentamicin"            : "bGentamicin", # Aminoglicosidos verde
            "Meropenem"             : "cMeropenem", # Carbapenemicos rojo
            "Ertapenem"             : "cErtapenem", # Carbapenemicos rojo
            "Cefixime"              : "dCefixime", # Cefalosporinas rosa
            "Ceftazidime"           : "dCeftazidime", # Cefalosporinas rosa
            "Ceftriaxone"           : "dCeftriaxone", # Cefalosporinas rosa
            "Colistin"              : "eColistin", # Polimixinas
            "Polymyxin"             : "ePolymyxin", # Polimixinas
            "Ciprofloxacin"         : "fCiprofloxacin", # Quinolonas burdeos
            "Levofloxacin"          : "fLevofloxacin", # Quinolonas burdeos
            "Fosfomycin"            : "gFosfomycin", # Sin grupo (grande) morado
            "Tigecycline"           : "gTigecycline", # Sin grupo (grande) morado
            "Tmp/smx"               : "hTmp/smx", # Sin grupo (pequeño) marrón
            "Meropenem-varbobactam" : "hMeropenem-varbobactam", # Sin grupo (pequeño) marrón
            "Metronidazole"         : "hMetronidazole", # Sin grupo (pequeño) marrón
            "Daptomycin"            : "hDaptomycin", # Sin grupo (pequeño) marrón
            "Linezolid"             : "hLinezolid",  # Sin grupo (pequeño) marrón
            "None"                  : "0None"}

    palette = {"zCaz-avi"               : "#f9be00", # CAZAVI   amarrillo
               "aInactive"              : "#555555", # Inactivo gris
               "bAmikacin"              : "#538136", # Aminoglicosidos verde
               "bGentamicin"            : "#a6d28e", # Aminoglicosidos verde
               "cMeropenem"             : "#ff1d20", # Carbapenemicos rojo
               "cErtapenem"             : "#9b0002", # Carbapenemicos rojo
               "dCefixime"              : "#af01b1", # Cefalosporinas rosa
               "dCeftazidime"           : "#ff70ff", # Cefalosporinas rosa
               "dCeftriaxone"           : "#febafd", # Cefalosporinas rosa
               "eColistin"              : "#01b0ee", # Polimixinas azul
               "ePolymyxin"             : "#77dbff", # Polimixinas azul
               "fCiprofloxacin"         : "#4a00d8", # Quinolonas burdeos
               "fLevofloxacin"          : "#6462d0", # Quinolonas burdeos
               "gFosfomycin"            : "#00ff04", # Sin grupo (grande) morado
               "gTigecycline"           : "#67fecb", # Sin grupo (grande) morado
               "hTmp/smx"               : "#ffd4b5", # Sin grupo (pequeño) marrón
               "hMeropenem-varbobactam" : "#fd7e25", # Sin grupo (pequeño) marrón
               "hMetronidazole"         : "#fda86d", # Sin grupo (pequeño) marrón
               "hDaptomycin"            : "#5c2400", # Sin grupo (pequeño) marrón
               "hLinezolid"             : "#d05503",  # Sin grupo (pequeño) marrón
               "0None"                  : "#999999"}


    names = list(palette.keys())


    maximum = []

    allAtbs = set()
    for patient in info:
        for step in patient.translateTreatmentLine(days = 30, start = patient.bloodculture):
            maximum.append(len(step))
            for atb in step:
                allAtbs.add(mask[atb])


    toDel = set()
    for atb in palette:
        if atb not in allAtbs:
            toDel.add(atb)

    for atb in toDel:
        del palette[atb]
        del mask[atb[1:]]

    inverseMask = {mask[i] : i for i in mask}

    maximum = max(maximum)


    buffer = pd.DataFrame({"Pt"   : [0] * len(allAtbs),
                           "Atb"  : [atb for atb in allAtbs],
                           "Step" : [0] * len(allAtbs)})


    res = {}

    for i in range(1, maximum + 1):
        for j in range(i):
            try:
                res[i][j] = pd.DataFrame(columns = ["Pt", "Atb", "Step"])
            except KeyError:
                res[i] = {j : pd.DataFrame(columns = ["Pt", "Atb", "Step"])}


    # Ordena pacientes de menos cambios a más cambios
    info.sort(key = lambda x: len(x.translateTreatmentLine(days = days,
                                                           start = x.bloodculture)), reverse = True)

    for indexPt, patient in enumerate(info):

        for indexStep, step in enumerate(patient.translateTreatmentLine(days = days, start = patient.bloodculture)):

            step = [mask[atb] for atb in step]
            step.sort(reverse = True)

            for n, atb in enumerate(step):

                if n != 0:

                    primeWidth = height / len(step) * (len(step) - n)

                    floor = indexStep - (height / 2)

                    primeCenter = floor + (primeWidth / 2)

                else:
                    primeCenter = indexStep

                res[len(step)][n].loc[len(res[len(step)][n].index)] = [indexPt,
                                                                       atb,
                                                                       primeCenter]

    plot = ["ggplot(aes('Pt', 'Step'))",
            "aes(fill = 'Atb')",
            "geom_tile(buffer, aes(width = width, height = height))"]



    for i in res:
        for j in res[i]:

            if j == 0:
                plot.append(f"geom_tile(res[{i}][{j}], aes(width = width, height = height))")

            else:
                plot.append(f"geom_tile(res[{i}][{j}], aes(width = width, height = height / {i} * {i - j}))")

    paco = {i : palette[i] for i in sorted(palette)}

    pepe = [inverseMask[i] for i in sorted(palette)]

    plot.append("scale_fill_manual(paco, labels = pepe)")
    plot.append("coord_equal(expand = False)")
    plot.append("theme_prism()")
    plot.append("theme(legend_position = (0.75, 0.7), legend_direction = 'horizontal')")
    plot.append("theme(axis_text_x = element_blank())")
    plot.append("theme(axis_ticks_x = element_blank())")
    plot.append(f"labs(title = '{title}')")
    plot.append("scale_y_continuous(name = 'Treatment Changes')")
    plot.append("scale_x_continuous(name = 'Patients')")
    plot.append("theme(figure_size = (20, 20))")


    return eval(" + ".join(plot))