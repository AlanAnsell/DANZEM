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

    
    def test_month_range(self):
        expected = [(2013, 2)]
        actual = data.MonthRange((2013, 2), (2013, 2))
        self.assertEqual(expected, actual)

        expected = [(2017, m) for m in range(3, 8)]
        actual = data.MonthRange((2017, 3), (2017, 7))
        self.assertEqual(expected, actual)

        expected = ([(2015, m) for m in range(7, 13)] +
                    [(2016, m) for m in range(1, 6)])
        actual = data.MonthRange((2015, 7), (2016, 5))
        self.assertEqual(expected, actual)

        expected = ([(2015, m) for m in range(7, 13)] +
                    [(2016, m) for m in range(1, 13)] +
                    [(2017, m) for m in range(1, 8)])
        actual = data.MonthRange((2015, 7), (2017, 7))
        self.assertEqual(expected, actual)

    
    def test_day_range(self):
        expected = [(2018, 2, d) for d in range(2, 21)]
        actual = data.DayRange((2018, 2, 2), (2018, 2, 20))
        self.assertEqual(expected, actual)

        expected = ([(2018, 2, d) for d in range(5, 29)] +
                    [(2018, 3, d) for d in range(1, 22)])
        actual = data.DayRange((2018, 2, 5), (2018, 3, 21))
        self.assertEqual(expected, actual)

        expected = ([(2017, 11, d) for d in range(5, 31)] +
                    [(2017, 12, d) for d in range(1, 32)] +
                    [(2018, 1, d) for d in range(1, 32)] +
                    [(2018, 2, d) for d in range(1, 29)] +
                    [(2018, 3, d) for d in range(1, 2)])
        actual = data.DayRange((2017, 11, 5), (2018, 3, 1))
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
