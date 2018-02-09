import unittest
from ..src import update


class TestUpdate(unittest.TestCase):

    def test_month_range(self):
        expected = [(2013, 2)]
        actual = update.MonthRange((2013, 2), (2013, 2))
        self.assertEqual(expected, actual)

        expected = [(2017, m) for m in range(3, 8)]
        actual = update.MonthRange((2017, 3), (2017, 7))
        self.assertEqual(expected, actual)

        expected = ([(2015, m) for m in range(7, 13)] +
                    [(2016, m) for m in range(1, 6)])
        actual = update.MonthRange((2015, 7), (2016, 5))
        self.assertEqual(expected, actual)

        expected = ([(2015, m) for m in range(7, 13)] +
                    [(2016, m) for m in range(1, 13)] +
                    [(2017, m) for m in range(1, 8)])
        actual = update.MonthRange((2015, 7), (2017, 7))
        self.assertEqual(expected, actual)

    def test_day_range(self):
        expected = [(2018, 2, d) for d in range(2, 21)]
        actual = update.DayRange((2018, 2, 2), (2018, 2, 20))
        self.assertEqual(expected, actual)

        expected = ([(2018, 2, d) for d in range(5, 29)] +
                    [(2018, 3, d) for d in range(1, 22)])
        actual = update.DayRange((2018, 2, 5), (2018, 3, 21))
        self.assertEqual(expected, actual)

        expected = ([(2017, 11, d) for d in range(5, 31)] +
                    [(2017, 12, d) for d in range(1, 32)] +
                    [(2018, 1, d) for d in range(1, 32)] +
                    [(2018, 2, d) for d in range(1, 29)] +
                    [(2018, 3, d) for d in range(1, 2)])
        actual = update.DayRange((2017, 11, 5), (2018, 3, 1))
        self.assertEqual(expected, actual)



if __name__ == '__main__':
    unittest.main()
