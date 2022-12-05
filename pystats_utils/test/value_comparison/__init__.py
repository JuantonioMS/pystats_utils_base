from pystats_utils.test import Test


class ValueComparison(Test):
    pass


from pystats_utils.test.value_comparison.mann_whitney_u_test import MannWhitneyUTest
from pystats_utils.test.value_comparison.t_test import StudentTTest
from pystats_utils.test.value_comparison.welch_test import WelchTest
from pystats_utils.test.value_comparison.wilcoxon_signed_rank_test import WilocoxonSignedRankTest
from pystats_utils.test.value_comparison.roc_analysis import RocAnalysis