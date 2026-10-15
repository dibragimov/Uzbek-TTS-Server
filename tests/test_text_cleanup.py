import json
import unittest
from pathlib import Path

from navoiy_server.text_cleanup import clean, looks_russian

RC4 = Path(__file__).resolve().parents[1] / "navoiy_server" / "rc4"


class Cleanup(unittest.TestCase):
    def test_rc4_telecom20_corpus_is_untouched(self):
        """The 20 validated sentences must reach the normalizer exactly as before."""
        rows = json.loads((RC4 / "telecom20_expected_normalized.json").read_text())
        self.assertEqual(len(rows), 20)
        for row in rows:
            self.assertEqual(clean(row["raw"]), row["raw"], row["label"])

    def test_telecom_and_uzbek_specials_survive(self):
        for text in ("Balansni tekshirish uchun *102# raqamini tering.",
                     "Qo'ng'iroqni yo'naltirish: **21*901234567#",
                     "O`zbekiston, g`oya, to`g`ri",       # backtick apostrophes
                     "Oʻzbekiston, gʻoya, toʻgʻri",       # modifier letter turned comma
                     "Narxi 1 250,50 so'm, 10 GB + 500 daqiqa.",
                     "Sana: 15.10.2026, vaqt 14:30.",
                     "Tarif — eng arzon variant.",
                     "Kod: 4827"):
            self.assertEqual(clean(text), text)

    def test_llm_formatting_removed(self):
        cases = {
            "**Rahmat!** Sizga yordam beraman.": "Rahmat! Sizga yordam beraman.",
            "## Tariflar\n- Start: 50 000 so'm\n- Pro: 90 000 so'm": "Tariflar Start: 50 000 so'm Pro: 90 000 so'm",
            "1. Ilovani oching\n2. Profilga kiring": "Ilovani oching Profilga kiring",
            "Rahmat! 😊👍": "Rahmat!",
            "Batafsil: [shu yerda](https://beeline.uz/tarif)": "Batafsil: shu yerda",
            "__muhim__ ~~eski~~ narx": "muhim eski narx",
            "Salom   \n\n  dunyo": "Salom dunyo",
            "🇺🇿 Xush kelibsiz": "Xush kelibsiz",
        }
        for raw, expected in cases.items():
            self.assertEqual(clean(raw), expected, raw)

    def test_russian_detection(self):
        self.assertTrue(looks_russian("Ваш баланс 25 000 сумов."))
        self.assertFalse(looks_russian("Раҳмат, сизга ёрдам бераман."))   # Cyrillic Uzbek
        self.assertFalse(looks_russian("Rahmat, sizga yordam beraman."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
