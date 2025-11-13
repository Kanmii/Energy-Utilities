
import unittest
from Solar_Capstone.simplified_app import calculate_requirement_from_daily_kwh, recommend_panels

class TestSimplifiedApp(unittest.TestCase):

    def test_calculate_requirement_from_daily_kwh(self):
        # Test with default values
        self.assertAlmostEqual(calculate_requirement_from_daily_kwh(12.0), 2.83636363636)
        # Test with custom sun hours
        self.assertAlmostEqual(calculate_requirement_from_daily_kwh(12.0, sun_hours=4.0), 3.9)
        # Test with zero sun hours (should use default)
        self.assertAlmostEqual(calculate_requirement_from_daily_kwh(12.0, sun_hours=0), 2.83636363636)

    def test_recommend_panels(self):
        # Test with a 3kW system
        recommendations = recommend_panels(3.0)
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]['panel_watt'], 400)
        self.assertEqual(recommendations[0]['count'], 8)
        self.assertEqual(recommendations[1]['panel_watt'], 350)
        self.assertEqual(recommendations[1]['count'], 9)
        self.assertEqual(recommendations[2]['panel_watt'], 300)
        self.assertEqual(recommendations[2]['count'], 10)

if __name__ == '__main__':
    unittest.main()
