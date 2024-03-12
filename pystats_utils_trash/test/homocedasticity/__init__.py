from pystats_utils.test import Test


class Homocedasticity(Test):
    pass


from pystats_utils.test.homocedasticity.bartlett_test import BartlettTest
from pystats_utils.test.homocedasticity.brown_forsyth_test import BrownForsythTest
from pystats_utils.test.homocedasticity.f_test import FTest
from pystats_utils.test.homocedasticity.fligner_killeen_test import FlignerTest
from pystats_utils.test.homocedasticity.levene_test import LeveneTest