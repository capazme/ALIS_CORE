"""Factory for generating Italian legal article test data."""

import random
from datetime import date, timedelta


_TIPI_ATTO = [
    ("legge", "stato"),
    ("decreto.legislativo", "stato"),
    ("decreto.legge", "stato"),
    ("codice.civile", "stato"),
    ("codice.penale", "stato"),
    ("costituzione", "stato"),
]

_RUBRICHE = [
    "Disposizioni generali",
    "Delle obbligazioni in generale",
    "Della responsabilita' civile",
    "Del contratto in generale",
    "Dei delitti contro la persona",
    "Dei diritti e doveri dei cittadini",
    "Norme transitorie e finali",
    "Della proprieta'",
    "Del possesso",
    "Dell'adempimento delle obbligazioni",
]

_SAMPLE_TEXTS = [
    "Il debitore che non esegue esattamente la prestazione dovuta e' tenuto al risarcimento del danno.",
    "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno.",
    "Non e' punibile chi ha commesso il fatto per esservi stato costretto dalla necessita' di difendere un diritto proprio od altrui.",
    "Tutti i cittadini hanno pari dignita' sociale e sono eguali davanti alla legge.",
    "Il contratto e' l'accordo di due o piu' parti per costituire, regolare o estinguere tra loro un rapporto giuridico patrimoniale.",
]


def create_article(**overrides) -> dict:
    """Create an Italian legal article test dict.

    Returns dict with: urn, tipo_atto, data, numero_atto, numero_articolo,
    rubrica, text.

    URN format: urn:nir:stato:{tipo_atto}:{data};{numero_atto}~art{numero}
    """
    tipo_atto, autorita = random.choice(_TIPI_ATTO)
    anno = random.randint(1942, 2025)
    mese = random.randint(1, 12)
    giorno = random.randint(1, 28)
    data_atto = date(anno, mese, giorno)
    numero_atto = str(random.randint(1, 500))
    numero_articolo = str(random.randint(1, 2500))

    urn = f"urn:nir:{autorita}:{tipo_atto}:{data_atto.isoformat()};{numero_atto}~art{numero_articolo}"

    defaults = {
        "urn": urn,
        "tipo_atto": tipo_atto,
        "data": data_atto.isoformat(),
        "numero_atto": numero_atto,
        "numero_articolo": numero_articolo,
        "rubrica": random.choice(_RUBRICHE),
        "text": random.choice(_SAMPLE_TEXTS),
    }
    defaults.update(overrides)
    return defaults
