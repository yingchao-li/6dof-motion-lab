import unittest

from i18n import LANGUAGES, TEXTS, translate


class TranslationTests(unittest.TestCase):
    def test_every_entry_has_all_three_languages(self):
        expected = set(LANGUAGES.values())
        for key, translations in TEXTS.items():
            self.assertEqual(expected, set(translations), key)
            self.assertTrue(all(value.strip() for value in translations.values()), key)

    def test_english_is_the_default(self):
        self.assertEqual(translate("trajectory_tab"), "Trajectory")

    def test_interpolation_works_in_all_languages(self):
        self.assertIn("3 clusters", translate("cluster_summary", "en", clusters=3, noise=0.1))
        self.assertIn("3 个簇", translate("cluster_summary", "zh", clusters=3, noise=0.1))
        self.assertIn("3 Cluster", translate("cluster_summary", "de", clusters=3, noise=0.1))


if __name__ == "__main__":
    unittest.main()
