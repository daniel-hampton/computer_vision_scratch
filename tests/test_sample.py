"""
Example test file
"""

import unittest

from .context import sample


class MyTest(unittest.TestCase):
    def test_words(self):
        words = "My String"
        self.assertEqual(words, sample.myprint(words), msg="hello there")


def test_myfunc_adds_2():
    assert sample.myfunc(3) == 5


if __name__ == '__main__':
    unittest.main()
