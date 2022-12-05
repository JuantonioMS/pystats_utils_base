from . import similarity, testDataframe

import pytest
from pathlib import Path

import rpy2.robjects as robj
robj.r("\n".join([f"setwd('{Path(__file__).parent}')",
                  f"df <- read.csv('../database/database_test_r.csv')"]))

variables = [("num1"), ("num2"), ("num3"),
             ("num4"), ("num5"), ("num6"),
             ("num7"), ("num8"), ("num9")]



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_agostino(targetVariable):

    from pystats_utils.test.normality import AgostinoTest

    result = AgostinoTest(dataframe = testDataframe,
                          classVariable = "group",
                          targetVariable = targetVariable).run()

    robj.r("library(fBasics)")
    robj.r(f"pvalue1 <- dagoTest(df[df$group == 'g1',]${targetVariable})@test$p.value")
    robj.r(f"pvalue2 <- dagoTest(df[df$group == 'g2',]${targetVariable})@test$p.value")
    robj.r(f"pvalue3 <- dagoTest(df[df$group == 'g3',]${targetVariable})@test$p.value")

    assert similarity(result.pvalue["g1"],
                      robj.r("pvalue1")[0],
                      decimal = 1e-10)
    assert similarity(result.pvalue["g2"],
                      robj.r("pvalue2")[0],
                      decimal = 1e-10)
    assert similarity(result.pvalue["g3"],
                      robj.r("pvalue3")[0],
                      decimal = 1e-10)



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_kolmogorovSmirnov(targetVariable):

    from pystats_utils.test.normality import KolmogorovSmirnovTest

    result = KolmogorovSmirnovTest(dataframe = testDataframe,
                                   classVariable = "group",
                                   targetVariable = targetVariable).run()

    robj.r("library(fBasics)")
    robj.r(f"pvalue1 <- ksnormTest(df[df$group == 'g1',]${targetVariable})@test$p.value")
    robj.r(f"pvalue2 <- ksnormTest(df[df$group == 'g2',]${targetVariable})@test$p.value")
    robj.r(f"pvalue3 <- ksnormTest(df[df$group == 'g3',]${targetVariable})@test$p.value")

    assert similarity(result.pvalue["g1"],
                      robj.r("pvalue1")[0],
                      decimal = 1e-10)
    assert similarity(result.pvalue["g2"],
                      robj.r("pvalue2")[0],
                      decimal = 1e-10)
    assert similarity(result.pvalue["g3"],
                      robj.r("pvalue3")[0],
                      decimal = 1e-10)