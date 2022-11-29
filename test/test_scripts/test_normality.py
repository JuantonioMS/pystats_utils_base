from . import similarity, testDataframe

import pytest


@pytest.mark.parametrize(('targetVariable',                   'pvalue'),
                         [("Body_mass_index",                 0.0),
                          ("Health_scale",                    2.088412e-64),
                          ("Days_bad_mental_health_pre30d",   0.0),
                          ("Days_bad_physical_health_pre30d", 0.0),
                          ("Age_scale",                       0.0),
                          ("Education_level",                 0.0),
                          ("Income_scale",                    0.0)])
def test_agostino(targetVariable, pvalue):

    from pystats_utils.test.normality import AgostinoTest

    result = AgostinoTest(dataframe = testDataframe,
                          targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)


@pytest.mark.parametrize(('targetVariable',                   'pvalue1',     'pvalue2',   'pvalue3'),
                         [("Body_mass_index",                 5.327007e-77,  0.0,         0.0),
                          ("Health_scale",                    0.02271217,    9.36593e-66, 6.154323e-87),
                          ("Days_bad_mental_health_pre30d",   1.120289e-159, 0.0,         0.0),
                          ("Days_bad_physical_health_pre30d", 4.09042e-79,   0.0,         0.0),
                          ("Age_scale",                       0.0,           0.0,         0.0),
                          ("Education_level",                 0.0,           0.0,         0.0),
                          ("Income_scale",                    1.404914e-09,  0.0,         0.0)])
def test_agostinoGroups(targetVariable, pvalue1, pvalue2, pvalue3):

    from pystats_utils.test.normality import AgostinoTest

    result = AgostinoTest(dataframe = testDataframe,
                          classVariable = "Diabetes",
                          targetVariable = targetVariable).run()

    assert similarity(result.pvalue["no"],     pvalue1, decimal = 0.05)
    assert similarity(result.pvalue["type_1"], pvalue2, decimal = 0.05)
    assert similarity(result.pvalue["type_2"], pvalue3, decimal = 0.05)


@pytest.mark.parametrize(('targetVariable',                   'pvalue'),
                         [("Body_mass_index",                 0.0),
                          ("Health_scale",                    0.0),
                          ("Days_bad_mental_health_pre30d",   0.0),
                          ("Days_bad_physical_health_pre30d", 0.0),
                          ("Age_scale",                       0.0),
                          ("Education_level",                 0.0),
                          ("Income_scale",                    0.0)])
def test_kolmogorovSmirnov(targetVariable, pvalue):

    from pystats_utils.test.normality import KolmogorovSmirnovTest

    result = KolmogorovSmirnovTest(dataframe = testDataframe,
                                   targetVariable = targetVariable).run()

    assert similarity(result.pvalue, pvalue)


@pytest.mark.parametrize(('targetVariable',                   'pvalue1',     'pvalue2', 'pvalue3'),
                         [("Body_mass_index",                 2.118449e-66,  0.0,       0.0),
                          ("Health_scale",                    2.290709e-299, 0.0,       0.0),
                          ("Days_bad_mental_health_pre30d",   0.0,           0.0,       0.0),
                          ("Days_bad_physical_health_pre30d", 0.0,           0.0,       0.0),
                          ("Age_scale",                       2.93572e-114,  0.0,       0.0),
                          ("Education_level",                 7.260656e-285, 0.0,       0.0),
                          ("Income_scale",                    4.950622e-145, 0.0,       0.0)])
def test_kolmogorovSmirnovGroups(targetVariable, pvalue1, pvalue2, pvalue3):

    from pystats_utils.test.normality import KolmogorovSmirnovTest

    result = KolmogorovSmirnovTest(dataframe = testDataframe,
                                   classVariable = "Diabetes",
                                   targetVariable = targetVariable).run()

    assert similarity(result.pvalue["no"],     pvalue1)
    assert similarity(result.pvalue["type_1"], pvalue2)
    assert similarity(result.pvalue["type_2"], pvalue3)