from . import testDataframe

from pathlib import Path

import rpy2.robjects as robj
robj.r("\n".join([f"setwd('{Path(__file__).parent}')",
                  f"df <- read.csv('../database/database_test_r.csv')"]))

import pytest

def test_bivariantTable():
    
    from pystats_utils.pipeline import BivariantTable
    
    result = BivariantTable(dataframe = testDataframe,
                            classVariable = "group").run()

    assert result.shape == (30, 10)
    
    
    
@pytest.mark.parametrize(('groupExcluded'),
                         [("g1"), ("g2"), ("g3")])
def test_logisticExploration(groupExcluded):
    
    from pystats_utils.pipeline import LogisticExploration
    
    data = testDataframe[testDataframe["group"] != groupExcluded]
    
    result = LogisticExploration(dataframe = data,
                                 classVariable = "group").run()

    assert result.shape == (24, 3)