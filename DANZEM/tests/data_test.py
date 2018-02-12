import unittest
from .. import data


class TestData(unittest.TestCase):

    def test_dst_begins(self):
        # Beginning of daylight saving 2013
        self.assertTrue(data.DSTBegins(2013, 9, 29))
        self.assertFalse(data.DSTBegins(2013, 9, 28))
        self.assertFalse(data.DSTBegins(2013, 9, 30))
        self.assertFalse(data.DSTBegins(2013, 4, 7))

        # Beginning of daylight saving 2014
        self.assertTrue(data.DSTBegins(2014, 9, 28))
        self.assertFalse(data.DSTBegins(2014, 9, 27))
        self.assertFalse(data.DSTBegins(2014, 9, 29))
        self.assertFalse(data.DSTBegins(2014, 4, 6))

        # Beginning of daylight saving 2015
        self.assertTrue(data.DSTBegins(2015, 9, 27))
        self.assertFalse(data.DSTBegins(2015, 9, 26))
        self.assertFalse(data.DSTBegins(2015, 9, 28))
        self.assertFalse(data.DSTBegins(2015, 4, 5))


    def test_dst_ends(self):
        # End of daylight saving 2012
        self.assertTrue(data.DSTEnds(2013, 4, 7))
        self.assertFalse(data.DSTEnds(2013, 4, 6))
        self.assertFalse(data.DSTEnds(2013, 4, 8))
        self.assertFalse(data.DSTEnds(2013, 9, 29))

        # End of daylight saving 2013
        self.assertTrue(data.DSTEnds(2014, 4, 6))
        self.assertFalse(data.DSTEnds(2014, 4, 5))
        self.assertFalse(data.DSTEnds(2014, 4, 7))
        self.assertFalse(data.DSTEnds(2014, 9, 28))

        # End of daylight saving 2014
        self.assertTrue(data.DSTEnds(2015, 4, 5))
        self.assertFalse(data.DSTEnds(2015, 4, 4))
        self.assertFalse(data.DSTEnds(2015, 4, 6))
        self.assertFalse(data.DSTEnds(2015, 9, 27))

if __name__ == '__main__':
    unittest.main()
