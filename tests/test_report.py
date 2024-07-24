import unittest
from rurage.report import RAGEReport

class TestReport(unittest.TestCase):
    def test_ragereport(self):
        report = RAGEReport(report_name="Test Report")
        self.assertEqual(report.report_name, "Test Report")

if __name__ == "__main__":
    unittest.main()
