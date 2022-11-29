from . import similarity, testDataframe

import pytest


@pytest.mark.parametrize(('targetVariable',                   'pvalue'),
                         [("Body_mass_index",                 1.531363e-19),
                          ("Health_scale",                    1.510035e-22),
                          ("Days_bad_mental_health_pre30d",   1.859716e-07),
                          ("Days_bad_physical_health_pre30d", 1.776008e-07),
                          ("Age_scale",                       1.877844e-08),
                          ("Education_level",                 1.977738e-06),
                          ("Income_scale",                    6.705768e-08)])
def test_mannWhitneyU(targetVariable, pvalue):

    from pystats_utils.test.value_comparison import MannWhitneyUTest

    data = testDataframe[testDataframe.Diabetes != "type_2"]

    result = MannWhitneyUTest(dataframe = data,
                              classVariable = "Diabetes",
                              targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)


@pytest.mark.parametrize(('targetVariable',                   'pvalue'),
                         [("Body_mass_index",                 2.139538e-19),
                          ("Health_scale",                    5.453835e-24),
                          ("Days_bad_mental_health_pre30d",   1.681294e-09),
                          ("Days_bad_physical_health_pre30d", 5.33664e-09),
                          ("Age_scale",                       1.226212e-08),
                          ("Education_level",                 1.590968e-06),
                          ("Income_scale",                    4.924477e-08)])
def test_studentT(targetVariable, pvalue):

    from pystats_utils.test.value_comparison import StudentTTest

    data = testDataframe[testDataframe.Diabetes != "type_2"]

    result = StudentTTest(dataframe = data,
                          classVariable = "Diabetes",
                          targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)


@pytest.mark.parametrize(('targetVariable',                   'pvalue'),
                         [("Body_mass_index",                 5.410242e-15),
                          ("Health_scale",                    4.963507e-20),
                          ("Days_bad_mental_health_pre30d",   1.019949e-05),
                          ("Days_bad_physical_health_pre30d", 7.722007e-06),
                          ("Age_scale",                       1.888631e-10),
                          ("Education_level",                 7.647992e-06),
                          ("Income_scale",                    6.948008e-07)])
def test_welch(targetVariable, pvalue):

    from pystats_utils.test.value_comparison import WelchTest

    data = testDataframe[testDataframe.Diabetes != "type_2"]

    result = WelchTest(dataframe = data,
                       classVariable = "Diabetes",
                       targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)


#  TODO Mann_Whitney