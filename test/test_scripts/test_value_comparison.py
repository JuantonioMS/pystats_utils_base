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
def test_mannWhitneyU(targetVariable):

    from pystats_utils.test.value_comparison import MannWhitneyUTest

    result = MannWhitneyUTest(dataframe = testDataframe,
                              classVariable = "group",
                              targetVariable = targetVariable).run()

    robj.r(f"pvalue1 <- wilcox.test(df[df$group == 'g1',]${targetVariable}, df[df$group == 'g2',]${targetVariable})$p.value")
    robj.r(f"pvalue2 <- wilcox.test(df[df$group == 'g1',]${targetVariable}, df[df$group == 'g3',]${targetVariable})$p.value")
    robj.r(f"pvalue3 <- wilcox.test(df[df$group == 'g2',]${targetVariable}, df[df$group == 'g3',]${targetVariable})$p.value")

    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_studentT(targetVariable):

    from pystats_utils.test.value_comparison import StudentTTest

    result = StudentTTest(dataframe = testDataframe,
                          classVariable = "group",
                          targetVariable = targetVariable).run()

    robj.r(f"pvalue1 <- t.test(df[df$group == 'g1',]${targetVariable}, df[df$group == 'g2',]${targetVariable}, var.equal = TRUE)$p.value")
    robj.r(f"pvalue2 <- t.test(df[df$group == 'g1',]${targetVariable}, df[df$group == 'g3',]${targetVariable}, var.equal = TRUE)$p.value")
    robj.r(f"pvalue3 <- t.test(df[df$group == 'g2',]${targetVariable}, df[df$group == 'g3',]${targetVariable}, var.equal = TRUE)$p.value")

    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_welch(targetVariable):

    from pystats_utils.test.value_comparison import WelchTest

    result = WelchTest(dataframe = testDataframe,
                       classVariable = "group",
                       targetVariable = targetVariable).run()

    robj.r(f"pvalue1 <- t.test(df[df$group == 'g1',]${targetVariable}, df[df$group == 'g2',]${targetVariable}, var.equal = FALSE)$p.value")
    robj.r(f"pvalue2 <- t.test(df[df$group == 'g1',]${targetVariable}, df[df$group == 'g3',]${targetVariable}, var.equal = FALSE)$p.value")
    robj.r(f"pvalue3 <- t.test(df[df$group == 'g2',]${targetVariable}, df[df$group == 'g3',]${targetVariable}, var.equal = FALSE)$p.value")

    assert similarity(result.pvalue["g1 vs. g2"],
                      robj.r("pvalue1")[0])
    assert similarity(result.pvalue["g1 vs. g3"],
                      robj.r("pvalue2")[0])
    assert similarity(result.pvalue["g2 vs. g3"],
                      robj.r("pvalue3")[0])


#  TODO wilcoxon