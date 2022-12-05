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
def test_bartlett(targetVariable):

    from pystats_utils.test.homocedasticity import BartlettTest

    result = BartlettTest(dataframe = testDataframe,
                          classVariable = "group",
                          targetVariable = targetVariable).run()

    robj.r("\n".join([f"pvalue1 <- bartlett.test("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g3',])$p.value"]))
    robj.r("\n".join([f"pvalue2 <- bartlett.test("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g2',])$p.value"]))
    robj.r("\n".join([f"pvalue3 <- bartlett.test("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g1',])$p.value"]))


    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_brownforsy(targetVariable):

    from pystats_utils.test.homocedasticity import BrownForsythTest

    result = BrownForsythTest(dataframe = testDataframe,
                              classVariable = "group",
                              targetVariable = targetVariable).run()

    robj.r("library(car)")
    robj.r("\n".join([f"pvalue1 <- leveneTest("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g3',],",
                      "center = 'median')$'Pr(>F)'"]))
    robj.r("\n".join([f"pvalue2 <- leveneTest("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g2',],",
                      "center = 'median')$'Pr(>F)'"]))
    robj.r("\n".join([f"pvalue3 <- leveneTest("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g1',],",
                      "center = 'median')$'Pr(>F)'"]))


    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_levene(targetVariable):

    from pystats_utils.test.homocedasticity import LeveneTest

    result = LeveneTest(dataframe = testDataframe,
                        classVariable = "group",
                        targetVariable = targetVariable).run()

    robj.r("library(car)")
    robj.r("\n".join([f"pvalue1 <- leveneTest("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g3',],",
                      "center = 'mean')$'Pr(>F)'"]))
    robj.r("\n".join([f"pvalue2 <- leveneTest("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g2',],",
                      "center = 'mean')$'Pr(>F)'"]))
    robj.r("\n".join([f"pvalue3 <- leveneTest("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g1',],",
                      "center = 'mean')$'Pr(>F)'"]))


    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_fligner(targetVariable):

    from pystats_utils.test.homocedasticity import FlignerTest

    result = FlignerTest(dataframe = testDataframe,
                         classVariable = "group",
                         targetVariable = targetVariable).run()

    robj.r("\n".join([f"pvalue1 <- fligner.test(",
                      f"formula = {targetVariable} ~ group,",
                      "data = df[df$group != 'g3',])$p.value"]))
    robj.r("\n".join([f"pvalue2 <- fligner.test(",
                      f"formula = {targetVariable} ~ group,",
                      "data = df[df$group != 'g2',])$p.value"]))
    robj.r("\n".join([f"pvalue3 <- fligner.test(",
                      f"formula = {targetVariable} ~ group,",
                      "data = df[df$group != 'g1',])$p.value"]))

    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_f(targetVariable):

    from pystats_utils.test.homocedasticity import FTest

    result = FTest(dataframe = testDataframe,
                   classVariable = "group",
                   targetVariable = targetVariable).run()

    robj.r("\n".join([f"pvalue1 <- var.test("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g3',],",
                      "alternative = 'two.sided')$p.value"]))
    robj.r("\n".join([f"pvalue2 <- var.test("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g2',],",
                      "alternative = 'two.sided')$p.value"]))
    robj.r("\n".join([f"pvalue3 <- var.test("
                      f"{targetVariable} ~ group,",
                      "data = df[df$group != 'g1',],",
                      "alternative = 'two.sided')$p.value"]))


    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0],
                      decimal = 1e-10)
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0],
                      decimal = 1e-10)
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0],
                      decimal = 1e-10)