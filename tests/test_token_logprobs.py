"""Testes sem rede para o parser e a nova página didática."""

from pathlib import Path
from types import SimpleNamespace
import math
import unittest

from streamlit.testing.v1 import AppTest

from token_logprobs import parse_chat_logprobs, parse_response_logprobs, readable_token


PAGE = Path(__file__).resolve().parents[1] / "pages" / "token_probabilities.py"


class TokenLogprobsTests(unittest.TestCase):
    def test_parser_calculates_probabilities_and_deduplicates_selected_token(self):
        candidate = lambda token, logprob: SimpleNamespace(token=token, logprob=logprob, bytes=None)
        selected = candidate(" mundo", math.log(0.6))
        selected.top_logprobs = [candidate(" mundo", math.log(0.6)), candidate(" dia", math.log(0.25))]
        response = SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[selected]))])

        steps = parse_chat_logprobs(response)

        self.assertAlmostEqual(steps[0]["probability"], 0.6)
        self.assertEqual(steps[0]["label"], "␠mundo")
        self.assertEqual(len(steps[0]["alternatives"]), 2)
        self.assertTrue(steps[0]["alternatives"][0]["selected"])

    def test_readable_token_handles_bytes_and_line_breaks(self):
        self.assertEqual(readable_token("", [240, 159, 140, 141]), "🌍")
        self.assertEqual(readable_token("\nOlá"), "↵Olá")

    def test_responses_parser_reads_output_text_blocks(self):
        item = SimpleNamespace(token="Olá", logprob=math.log(0.8), bytes=None, top_logprobs=[])
        block = SimpleNamespace(logprobs=[item])
        response = SimpleNamespace(output=[SimpleNamespace(content=[block])])
        self.assertAlmostEqual(parse_response_logprobs(response)[0]["probability"], 0.8)

    def test_page_loads_without_api_key(self):
        app = AppTest.from_file(str(PAGE), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("OPENAI_API_KEY" in item.value for item in app.info))
        self.assertEqual(next(item for item in app.selectbox if item.label == "API").value, "Chat Completions")
        models = next(item for item in app.selectbox if item.label == "Modelo")
        self.assertIn("gpt-4.1", models.options)
        self.assertIn("gpt-4o-mini", models.options)
        temperature = next(item for item in app.slider if item.label == "Temperatura")
        self.assertEqual(temperature.value, 1.0)


if __name__ == "__main__":
    unittest.main()
