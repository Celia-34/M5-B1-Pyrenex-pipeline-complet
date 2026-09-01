"""Tests statiques du formulaire frontend (services/frontend/html/index.html).

pytest n'exécute pas le JavaScript : on vérifie les CONTRATS du formulaire,
pas son comportement dynamique.
  - les 14 champs correspondent au schéma Pydantic LoanApplication ;
  - les <select> n'exposent que des modalités connues du modèle ;
  - l'appel API reste en chemin relatif (proxy nginx, pas de CORS).
"""
from __future__ import annotations

import importlib.util
import sys
import typing
from html.parser import HTMLParser
from pathlib import Path

import joblib
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_HTML = REPO_ROOT / "services" / "frontend" / "html" / "index.html"
MODEL_PATH = REPO_ROOT / "services" / "model" / "models" / "pyrenex_risk_v2.joblib"
SCHEMAS_PY = REPO_ROOT / "services" / "backend" / "app" / "schemas.py"

NUMERIC_FIELDS = {
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "revol_util",
}


class FormParser(HTMLParser):
    """Collecte les champs nommés et les options de chaque <select>."""

    def __init__(self) -> None:
        super().__init__()
        self.fields: dict[str, dict] = {}   # name -> attributs
        self.options: dict[str, set] = {}   # name du select -> valeurs
        self.sentinels: set[str] = set()    # selects avec placeholder vide
        self._current: str | None = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ("input", "select") and d.get("name"):
            self.fields[d["name"]] = d
            if tag == "select":
                self._current = d["name"]
                self.options[d["name"]] = set()
        elif tag == "option" and self._current:
            if d.get("value"):
                self.options[self._current].add(d["value"])
            elif "disabled" in d and "selected" in d:
                self.sentinels.add(self._current)

    def handle_endtag(self, tag):
        if tag == "select":
            self._current = None


@pytest.fixture(scope="module")
def html_text() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def form(html_text) -> FormParser:
    p = FormParser()
    p.feed(html_text)
    return p


@pytest.fixture(scope="module")
def schema():
    """LoanApplication chargé par chemin, sans polluer le package `app`."""
    spec = importlib.util.spec_from_file_location("backend_schemas", SCHEMAS_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backend_schemas"] = mod
    spec.loader.exec_module(mod)
    return mod.LoanApplication


@pytest.fixture(scope="module")
def encoder_categories() -> dict[str, set]:
    """Modalités réellement apprises par le OneHotEncoder du pipeline."""
    pre = joblib.load(MODEL_PATH).named_steps["preprocess"]
    for _, trans, cols in pre.transformers_:
        inner = trans.named_steps.values() if hasattr(trans, "named_steps") else [trans]
        for step in inner:
            if hasattr(step, "categories_"):
                return {c: set(cat) for c, cat in zip(cols, step.categories_)}
    pytest.fail("Aucun encodeur catégoriel trouvé dans le pipeline")


# --- Contrat formulaire <-> schéma Pydantic ---------------------------------

def test_les_14_champs_du_schema_sont_presents(form, schema):
    assert set(form.fields) == set(schema.model_fields)


def test_les_champs_numeriques_portent_le_marqueur_de_conversion(form):
    """Sans data-type="number", FormData enverrait des chaînes -> 422."""
    for name in NUMERIC_FIELDS:
        assert form.fields[name].get("data-type") == "number", name


def test_les_bornes_ne_sont_pas_dupliquees_en_html(form):
    """Choix d'architecture : la validation des bornes est SERVEUR uniquement.

    Pas de min/max sur les <input> : une borne recopiée en HTML est une
    duplication de la règle métier définie dans LoanApplication, qui dérive
    dès que le schéma change. Le navigateur laisse passer, Pydantic tranche,
    et le bloc 422 du script affiche le champ fautif.
    """
    for name in NUMERIC_FIELDS:
        attrs = form.fields[name]
        assert "min" not in attrs and "max" not in attrs, name


# --- Contrat formulaire <-> modèle ------------------------------------------

def test_les_options_correspondent_aux_modalites_du_modele(form, encoder_categories):
    """L'encodeur est en handle_unknown='ignore' : une modalité inconnue ne
    lève pas d'erreur, elle produit un vecteur nul et une prédiction fausse.
    Le formulaire est le seul garde-fou."""
    for field, expected in encoder_categories.items():
        assert form.options[field] == expected, field


def test_les_options_sont_acceptees_par_le_schema(form, schema):
    """Inclusion, pas égalité : le Literal de home_ownership est plus large
    (6 valeurs) que ce que le modèle connaît (4)."""
    hints = typing.get_type_hints(schema)
    for field, values in form.options.items():
        allowed = typing.get_args(hints[field])
        if allowed:  # les champs typés `str` libre n'ont rien à vérifier ici
            assert values <= set(allowed), field


def test_chaque_select_a_une_option_placeholder(form):
    """`required` sur un <select> n'agit que si l'option retenue a value="".
    Sans placeholder, la première option est choisie automatiquement."""
    assert form.sentinels == set(form.options)


# --- Contrat formulaire <-> réseau ------------------------------------------

def test_l_appel_api_reste_en_chemin_relatif(html_text):
    """nginx proxifie /api/ vers le backend : même origine, pas de CORS.
    Une URL absolue casserait dès que les ports changent."""
    assert "'/api/score'" in html_text or '"/api/score"' in html_text
    assert "http://localhost:8001" not in html_text
    assert "http://backend:8001" not in html_text


def test_aucun_todo_residuel(html_text):
    assert "TODO" not in html_text
