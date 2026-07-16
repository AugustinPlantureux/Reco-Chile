from __future__ import annotations

import ast
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def literal_translation_keys() -> set[str]:
    keys: set[str] = set()
    paths = [ROOT / "app.py", *sorted((ROOT / "sae_app").glob("*.py"))]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "t":
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                keys.add(first_arg.value)
    return keys


def translation_mapping() -> dict:
    tree = ast.parse((ROOT / "sae_app" / "i18n.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "TRANSLATIONS" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("TRANSLATIONS mapping not found")


class FamilyUiTests(unittest.TestCase):
    def test_every_literal_ui_string_has_a_spanish_translation(self) -> None:
        translations = translation_mapping()
        missing = literal_translation_keys() - set(translations["es"])
        self.assertEqual(missing, set())

    def test_initial_spanish_and_english_views_render_without_errors(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            app.title[0].value,
            "Revisa el riesgo de tu lista de preferencias SAE",
        )

        app.selectbox(key="language_selector_mtb").select("en").run(timeout=60)
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.title[0].value, "Review the risk of your SAE preference list")


if __name__ == "__main__":
    unittest.main()
