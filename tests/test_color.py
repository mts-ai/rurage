import unittest
from rurage.color import Color

class TestColor(unittest.TestCase):
    def test_stylify(self):
        self.assertEqual(Color.stylify("test", "BOLD"), "\033[1mtest\033[0m")

if __name__ == "__main__":
    unittest.main()
