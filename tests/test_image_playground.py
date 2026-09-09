"""Regressões da migração de imagens; respostas da API simuladas, sem cobranças."""

import base64
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest


PAGE = Path(__file__).resolve().parents[1] / "pages" / "playground_imagem.py"


def widget(elements, label):
    return next(element for element in elements if element.label == label)


class ImagePlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(PAGE), default_timeout=20)
        self.app.secrets["OPENAI_API_KEY"] = "test-only"
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_new_controls_and_jpeg_transparency(self):
        self.assertEqual(widget(self.app.selectbox, "Modelo de imagem").value, "GPT Image 2.5 Flare")
        quality = widget(self.app.select_slider, "Qualidade")
        self.assertIn("xhigh", quality.options)
        self.assertIn("max", quality.options)
        widget(self.app.selectbox, "Fundo").select("transparent").run()
        widget(self.app.selectbox, "Formato").select("jpeg").run()
        self.assertNotIn("transparent", widget(self.app.selectbox, "Fundo").options)
        self.assertFalse(self.app.exception)

    def test_generate_and_edit_use_selected_model_and_preserve_result(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (16, 16), (0, 128, 255, 128)).save(buffer, format="PNG")
        picture = base64.b64encode(buffer.getvalue()).decode()
        usage = SimpleNamespace(input_tokens_details=SimpleNamespace(text_tokens=10, image_tokens=20), output_tokens=100)
        event = SimpleNamespace(type="image_generation.completed", b64_json=picture, usage=usage)
        for name, model in [("GPT Image 2.5 Flare", "gpt-image-2.5-flare"), ("GPT Image 2.5 Sunburst", "gpt-image-2.5-sunburst")]:
            with self.subTest(model=model), patch("openai.OpenAI") as client_class:
                client = client_class.return_value
                client.images.generate.return_value = iter([event])
                widget(self.app.selectbox, "Modelo de imagem").select(name).run()
                widget(self.app.selectbox, "Fundo").select("transparent").run()
                widget(self.app.select_slider, "Qualidade").set_value("max").run()
                self.app.text_area[0].set_value("Um círculo azul").run()
                widget(self.app.button, "Gerar imagem").click().run()
                self.assertFalse(self.app.exception)
                params = client.images.generate.call_args.kwargs
                self.assertEqual(params["model"], model)
                self.assertEqual(params["quality"], "max")
                self.assertEqual(params["background"], "transparent")
                self.assertTrue(params["stream"])
                result = self.app.session_state["imagem_resultados"]
                self.assertEqual(result["modelo"], model)
                self.assertAlmostEqual(result["custo_tokens"], 0.00321)

                reference = MagicMock(name="reference")
                reference.name, reference.type = "reference.png", "image/png"
                reference.getvalue.return_value = buffer.getvalue()
                # Sem prévia de referências neste teste: st.image aceita bytes.
                client.images.edit.return_value = iter([event])
                with patch("streamlit.file_uploader", side_effect=lambda label, **kwargs: [reference] if label == "Imagens de referência" else None), patch("streamlit.image"):
                    widget(self.app.radio, "Modo").set_value("Editar / usar referências").run()
                    widget(self.app.button, "Editar imagem").click().run()
                    self.assertFalse(self.app.exception)
                    params = client.images.edit.call_args.kwargs
                    self.assertEqual(params["model"], model)
                    self.assertEqual(params["quality"], "max")
                    self.assertEqual(params["image"][0][0], "reference.png")
                widget(self.app.radio, "Modo").set_value("Gerar do zero").run()

        widget(self.app.selectbox, "Modelo de imagem").select("GPT Image 2.5 Flare").run()
        self.assertEqual(self.app.session_state["imagem_resultados"]["modelo"], "gpt-image-2.5-sunburst")

    def test_legacy_selection_resets_unsupported_options_and_routes_request(self):
        widget(self.app.selectbox, "Fundo").select("transparent").run()
        widget(self.app.select_slider, "Qualidade").set_value("max").run()
        widget(self.app.selectbox, "Modelo de imagem").select("GPT Image 2.0 (anterior)").run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("max", widget(self.app.select_slider, "Qualidade").options)
        self.assertNotIn("xhigh", widget(self.app.select_slider, "Qualidade").options)
        self.assertNotIn("transparent", widget(self.app.selectbox, "Fundo").options)
        with patch("openai.OpenAI") as client_class:
            client_class.return_value.images.generate.return_value = iter([])
            self.app.text_area[0].set_value("Um círculo azul").run()
            widget(self.app.button, "Gerar imagem").click().run()
            params = client_class.return_value.images.generate.call_args.kwargs
            self.assertEqual(params["model"], "gpt-image-2")
            self.assertIn(params["quality"], ["auto", "low", "medium", "high"])
            self.assertIn(params["background"], ["auto", "opaque"])
        widget(self.app.selectbox, "Modelo de imagem").select("GPT Image 2.5 Flare").run()
        self.assertIn("max", widget(self.app.select_slider, "Qualidade").options)
        self.assertIn("transparent", widget(self.app.selectbox, "Fundo").options)

    def test_verification_error_and_empty_stream_are_not_success(self):
        with patch("openai.OpenAI") as client_class:
            self.app.text_area[0].set_value("Um círculo azul").run()
            client_class.return_value.images.generate.side_effect = RuntimeError("Your organization must be verified to use the model")
            widget(self.app.button, "Gerar imagem").click().run()
            self.assertFalse(self.app.exception)
            self.assertTrue(any("verificação da organização" in warning.value for warning in self.app.warning))
            self.assertFalse(any("Concluído" in item.value for item in self.app.success))
            client_class.return_value.images.generate.side_effect = None
            client_class.return_value.images.generate.return_value = iter([])
            widget(self.app.button, "Gerar imagem").click().run()
            self.assertTrue(any("sem retornar uma imagem final" in item.value for item in self.app.code))
            self.assertFalse(any("Concluído" in item.value for item in self.app.success))


if __name__ == "__main__":
    unittest.main()
