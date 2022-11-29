from . import similarity, testDataframe

import pytest


@pytest.mark.parametrize(('targetVariable',                   'pvalue'),
                         [("Body_mass_index",                 0.02906807),
                          ("Health_scale",                    0.5735045),
                          ("Days_bad_mental_health_pre30d",   2.808665e-15),
                          ("Days_bad_physical_health_pre30d", 5.74834e-11),
                          ("Age_scale",                       0.0006726),
                          ("Education_level",                 0.2007997),
                          ("Income_scale",                    0.0730407)])
def test_bartlett(targetVariable, pvalue):

    from pystats_utils.test.homocedasticity import BartlettTest

    data = testDataframe[testDataframe.Diabetes != "type_2"]

    result = BartlettTest(dataframe = data,
                          classVariable = "Diabetes",
                          targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)


@pytest.mark.parametrize(('targetVariable',                   'pvalue1',    'pvalue2',    'pvalue3'),
                         [("Body_mass_index",                 0.02906807,   1.005764e-35, 0.03707809),
                          ("Health_scale",                    0.5735045,    0.05623369,   0.2216088),
                          ("Days_bad_mental_health_pre30d",   2.808665e-15, 2.596767e-61, 0.1606269),
                          ("Days_bad_physical_health_pre30d", 5.74834e-11,  3.507784e-94, 0.3443797),
                          ("Age_scale",                       0.0006726,    2.606134e-68, 0.001386579),
                          ("Education_level",                 0.2007997,    1.847491e-16, 0.1202821),
                          ("Income_scale",                    0.0730407,    8.811896e-06, 0.8785819)])
def test_bartlettGroups(targetVariable, pvalue1, pvalue2, pvalue3):

    from pystats_utils.test.homocedasticity import BartlettTest

    result = BartlettTest(dataframe = testDataframe,
                          classVariable = "Diabetes",
                          targetVariable = targetVariable).run()

    assert similarity(result.pvalue["no vs. type_1"], pvalue1)
    assert similarity(result.pvalue["no vs. type_2"], pvalue2)
    assert similarity(result.pvalue["type_1 vs. type_2"], pvalue3)
