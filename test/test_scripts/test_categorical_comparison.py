from . import similarity, testDataframe

import pytest
from pathlib import Path

import rpy2.robjects as robj
robj.r("\n".join([f"setwd('{Path(__file__).parent}')",
                  f"df <- read.csv('../database/database_test_r.csv')"]))

variables = [("cat1"), ("cat2"), ("cat3"),
             ("cat4"), ("cat5"), ("cat6")]



@pytest.mark.parametrize(('targetVariable'),
                         variables)
def test_pearsonChiSquare(targetVariable):

    from pystats_utils.test.categorical_comparison import PearsonChiSquareTest

    result = PearsonChiSquareTest(dataframe = testDataframe,
                                  classVariable = "group",
                                  targetVariable = targetVariable).run()

    robj.r(f"pvalue <- chisq.test(df${targetVariable}, df$group)$p.value")

    assert similarity(result.pvalue,
                      robj.r("pvalue")[0])