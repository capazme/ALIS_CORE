"""
MERL-T Sources Utils
====================

Utilità per gli scrapers: parsing URN, estrazione struttura, operazioni testo.

Componenti:
- norma: NormaVisitata, Modifica, TipoModifica, StoriaArticolo
- urn: generate_urn, parse_urn
- tree: NormTree, get_article_position, get_hierarchical_tree
- text: normalize_act_type, clean_text
- http: HTTP client utilities
- map: Mappature codici (BROCARDI_CODICI, etc.)
"""

from visualex.models.norma import (
    NormaVisitata,
    Modifica,
    TipoModifica,
    StoriaArticolo,
)
from visualex.utils.urngenerator import generate_urn
from visualex.utils.treextractor import (
    NormTree,
    get_article_position,
    get_hierarchical_tree,
)
from visualex.utils.text_op import normalize_act_type
from visualex.utils.map import BROCARDI_CODICI

__all__ = [
    # Norma models
    "NormaVisitata",
    "Modifica",
    "TipoModifica",
    "StoriaArticolo",
    # URN
    "generate_urn",
    # Tree
    "NormTree",
    "get_article_position",
    "get_hierarchical_tree",
    # Text
    "normalize_act_type",
    # Map
    "BROCARDI_CODICI",
]
