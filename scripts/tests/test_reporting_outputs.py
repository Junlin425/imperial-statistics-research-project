import unittest

from reporting_outputs import make_validity_row


class ValiditySummaryTests(unittest.TestCase):
    def test_row_reports_nominal_and_effective_acceptance_fractions(self):
        row = make_validity_row(
            "example bank",
            nominal_draws=100,
            valid_draws=80,
            accepted_draws=10,
        )

        self.assertEqual(row["invalid_draws"], 20)
        self.assertAlmostEqual(row["invalid_fraction"], 0.20)
        self.assertAlmostEqual(row["accepted_fraction_of_nominal"], 0.10)
        self.assertAlmostEqual(row["accepted_fraction_of_valid"], 0.125)


if __name__ == "__main__":
    unittest.main()
