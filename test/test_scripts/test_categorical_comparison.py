from . import similarity, testDataframe

import pytest


@pytest.mark.parametrize(('targetVariable',               'pvalue'),
                         [("High_BP",                     1.275011e-10),
                          ("High_cholesterol",            3.195678e-17),
                          ("Cholesterol_check_pre5y",     0.004967943),
                          ("Smoker",                      0.02880998),
                          ("Stroke",                      0.005356502),
                          ("Heart_disease_or_attack",     0.000409245),
                          ("Physical_activity_pre30d",    5.302342e-05),
                          ("Fruits_diet",                 0.006669877),
                          ("Veggies_diet",                0.0192896),
                          ("Heavy_alcohol_consump",       0.1202403),
                          ("Any_healthcare",              0.08981938),
                          ("No_money_for_doctors_pre12m", 0.03488492),
                          ("Difficulty_stairs",           3.697693e-14),
                          ("Sex",                         0.2190996)])
def test_pearsonChiSquare(targetVariable, pvalue):

    from pystats_utils.test.categorical_comparison import PearsonChiSquareTest

    data = testDataframe[testDataframe.Diabetes != "type_2"]

    result = PearsonChiSquareTest(dataframe = data,
                                  classVariable = "Diabetes",
                                  targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)