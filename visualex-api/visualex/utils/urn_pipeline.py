"""
URN Canonicalization Pipeline for Italian Legal Norms.

Implements parsing, validation, and canonicalization of URNs following
the NIR (Norme In Rete) standard as documented by Normattiva.

URN Format: urn:nir:stato:[tipo_atto]:[data];[numero]:[allegato]~art[N][ext][@orig|!vig=date]

Note: Normattiva URNs support precision only up to article level.
      Comma and lettera references are not supported by the URN standard.

Reference: https://www.normattiva.it/do/atto/export/help
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import structlog

log = structlog.get_logger()

__all__ = [
    'URNComponents',
    'parse_urn',
    'validate_urn',
    'canonicalize_urn',
    'urns_are_equivalent',
    'extract_article_info',
    'urn_to_normattiva_url',
    'dict_to_urn',
    'normalize_act_type_for_urn',
    'extension_to_number',
    'number_to_extension',
    'NORMATTIVA_BASE_URL',
    'URN_PREFIX',
    'LATIN_EXTENSIONS',
]

# Valid act types (normalized form with dots)
# Comprehensive list based on NORMATTIVA_URN_CODICI and common Italian legal acts
VALID_ACT_TYPES = {
    # Primary legislation
    'legge', 'decreto.legge', 'decreto.legislativo',
    'costituzione', 'legge.costituzionale',
    # Presidential/Government decrees
    'decreto.del.presidente.della.repubblica', 'regio.decreto',
    'decreto.ministeriale', 'decreto.del.presidente.del.consiglio.dei.ministri',
    # Historical/Special
    'relazione.e.regio.decreto',
    # Regional (for completeness, though Normattiva focuses on national)
    'legge.regionale',
}

# Latin ordinal extensions for article numbering
LATIN_EXTENSIONS = {
    '': 0, 'bis': 2, 'ter': 3, 'tris': 3, 'quater': 4,
    'quinquies': 5, 'quinques': 5, 'sexies': 6, 'septies': 7, 'octies': 8,
    'novies': 9, 'decies': 10, 'undecies': 11, 'duodecies': 12, 'terdecies': 13,
    'quaterdecies': 14, 'quindecies': 15, 'sexdecies': 16, 'septiesdecies': 17,
    'duodevicies': 18, 'undevicies': 19, 'vices': 20, 'vicessemel': 21,
    'vicesbis': 22, 'vicester': 23, 'vicesquater': 24, 'vicesquinquies': 25,
    'vicessexies': 26, 'vicessepties': 27, 'duodetricies': 28, 'undetricies': 29,
    'tricies': 30, 'triciessemel': 31, 'triciesbis': 32, 'triciester': 33,
    'triciesquater': 34, 'triciesquinquies': 35, 'triciessexies': 36,
    'triciessepties': 37, 'duodequadragies': 38, 'undequadragies': 39,
    'quadragies': 40, 'quadragiessemel': 41, 'quadragiesbis': 42,
    'quadragiester': 43, 'quadragiesquater': 44, 'quadragiesquinquies': 45,
    'quadragiessexies': 46, 'quadragiessepties': 47, 'duodequinquagies': 48,
    'undequinquagies': 49,
}

# Reverse lookup: number -> extension (for number_to_extension, O(1) instead of O(n))
# Uses canonical form (e.g., 'ter' not 'tris')
_NUMBER_TO_EXTENSION = {
    2: 'bis', 3: 'ter', 4: 'quater', 5: 'quinquies', 6: 'sexies',
    7: 'septies', 8: 'octies', 9: 'novies', 10: 'decies', 11: 'undecies',
    12: 'duodecies', 13: 'terdecies', 14: 'quaterdecies', 15: 'quindecies',
    16: 'sexdecies', 17: 'septiesdecies', 18: 'duodevicies', 19: 'undevicies',
    20: 'vices', 21: 'vicessemel', 22: 'vicesbis', 23: 'vicester',
    24: 'vicesquater', 25: 'vicesquinquies', 26: 'vicessexies', 27: 'vicessepties',
    28: 'duodetricies', 29: 'undetricies', 30: 'tricies', 31: 'triciessemel',
    32: 'triciesbis', 33: 'triciester', 34: 'triciesquater', 35: 'triciesquinquies',
    36: 'triciessexies', 37: 'triciessepties', 38: 'duodequadragies',
    39: 'undequadragies', 40: 'quadragies', 41: 'quadragiessemel',
    42: 'quadragiesbis', 43: 'quadragiester', 44: 'quadragiesquater',
    45: 'quadragiesquinquies', 46: 'quadragiessexies', 47: 'quadragiessepties',
    48: 'duodequinquagies', 49: 'undequinquagies',
}

# Normattiva URL prefix
NORMATTIVA_BASE_URL = "https://www.normattiva.it/uri-res/N2Ls?"

# URN prefix
URN_PREFIX = "urn:nir:stato:"


@dataclass
class URNComponents:
    """
    Parsed components of a NIR-compliant URN.

    Represents all extractable parts from a URN following the format:
    urn:nir:stato:[tipo_atto]:[data];[numero]:[allegato]~art[articolo][estensione][@originale|!vig=[data_versione]]
    """
    tipo_atto: str
    data: Optional[str] = None
    numero: Optional[str] = None
    allegato: Optional[str] = None
    articolo: Optional[str] = None
    estensione: Optional[str] = None
    versione: Optional[str] = None  # "originale" or "vigente"
    data_versione: Optional[str] = None

    # Validation metadata
    is_valid: bool = field(default=True, repr=False)
    validation_errors: list = field(default_factory=list, repr=False)

    def to_urn(self, include_url: bool = False) -> str:
        """
        Convert components back to canonical URN string.

        Args:
            include_url: If True, prepend Normattiva base URL

        Returns:
            Canonical URN string
        """
        parts = [URN_PREFIX, self.tipo_atto]

        if self.data:
            parts.append(f":{self.data}")

        if self.numero:
            parts.append(f";{self.numero}")

        if self.allegato:
            parts.append(f":{self.allegato}")

        if self.articolo:
            article_part = f"~art{self.articolo}"
            if self.estensione:
                article_part += self.estensione
            parts.append(article_part)

        if self.versione == "originale":
            parts.append("@originale")
        elif self.versione == "vigente":
            if self.data_versione:
                parts.append(f"!vig={self.data_versione}")
            else:
                parts.append("!vig=")

        urn = "".join(parts)

        if include_url:
            return NORMATTIVA_BASE_URL + urn
        return urn

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'tipo_atto': self.tipo_atto,
            'data': self.data,
            'numero': self.numero,
            'allegato': self.allegato,
            'articolo': self.articolo,
            'estensione': self.estensione,
            'versione': self.versione,
            'data_versione': self.data_versione,
            'is_valid': self.is_valid,
        }

    def canonical_key(self) -> str:
        """
        Generate a canonical key for duplicate detection.

        Returns normalized key ignoring version info, useful for
        detecting when different URNs refer to the same norm.
        """
        key_parts = [self.tipo_atto]
        if self.data:
            key_parts.append(self.data)
        if self.numero:
            key_parts.append(self.numero)
        if self.allegato:
            key_parts.append(f"all{self.allegato}")
        if self.articolo:
            art_key = f"art{self.articolo}"
            if self.estensione:
                art_key += self.estensione.lower()
            key_parts.append(art_key)
        return ":".join(key_parts)


# Regex patterns for URN parsing
# Pattern for the base URN (without URL prefix)
# Article format can be: ~art[N], ~art[N][ext], ~art[N][ext].[subpart], ~art[N].[subpart]
# Examples: ~art2, ~art16bis, ~art79octies.1, ~art314.15
URN_PATTERN = re.compile(
    r'^(?:https?://[^?]+\?)?'  # Optional URL prefix
    r'urn:nir:stato:'
    r'(?P<tipo_atto>[a-z.]+)'
    r'(?::(?P<data>\d{4}-\d{2}-\d{2}))?'
    r'(?:;(?P<numero>\d+))?'
    r'(?::(?P<allegato>\d+))?'
    r'(?:~art(?P<articolo>\d+)(?P<estensione>[a-z]*)(?P<subpart>\.[\d.]+)?)?'
    r'(?:(?P<originale>@originale)|(?P<vigente>!vig=)(?P<data_versione>\d{4}-\d{2}-\d{2})?)?'
    r'$',
    re.IGNORECASE
)

# Date validation pattern
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def parse_urn(urn: str) -> URNComponents:
    """
    Parse a URN string into its component parts.

    Args:
        urn: URN string (with or without Normattiva URL prefix)

    Returns:
        URNComponents with extracted values

    Examples:
        >>> parse_urn("urn:nir:stato:legge:2020-12-30;178")
        URNComponents(tipo_atto='legge', data='2020-12-30', numero='178', ...)

        >>> parse_urn("urn:nir:stato:regio.decreto:1942-03-16;262:2~art1453")
        URNComponents(tipo_atto='regio.decreto', ..., allegato='2', articolo='1453')
    """
    if not urn:
        return URNComponents(
            tipo_atto="",
            is_valid=False,
            validation_errors=["Empty URN"]
        )

    # Normalize: strip whitespace
    urn = urn.strip()

    match = URN_PATTERN.match(urn)

    if not match:
        log.warning("Failed to parse URN", urn=urn)
        return URNComponents(
            tipo_atto="",
            is_valid=False,
            validation_errors=[f"Invalid URN format: {urn}"]
        )

    groups = match.groupdict()

    # Determine version type
    versione = None
    if groups.get('originale'):
        versione = "originale"
    elif groups.get('vigente') is not None:
        versione = "vigente"

    # Handle article with subpart (e.g., 79octies.1 or 314.15)
    articolo = groups.get('articolo')
    estensione = groups.get('estensione') or None
    subpart = groups.get('subpart')  # e.g., ".1" or ".15"

    # If there's a subpart, append it to articolo (after extension)
    # This handles cases like art79octies.1 → articolo="79", estensione="octies", subpart=".1"
    # We combine estensione+subpart into estensione for simplicity
    if subpart:
        if estensione:
            estensione = estensione + subpart
        else:
            # No extension, just subpart like art314.15
            articolo = articolo + subpart

    components = URNComponents(
        tipo_atto=groups['tipo_atto'].lower(),
        data=groups.get('data'),
        numero=groups.get('numero'),
        allegato=groups.get('allegato'),
        articolo=articolo,
        estensione=estensione if estensione else None,
        versione=versione,
        data_versione=groups.get('data_versione'),
    )

    # Run validation
    errors = _validate_components(components)
    if errors:
        components.is_valid = False
        components.validation_errors = errors

    log.debug("Parsed URN", urn=urn, components=components.to_dict())
    return components


def validate_urn(urn: str) -> Tuple[bool, list]:
    """
    Validate a URN string against NIR standard.

    Args:
        urn: URN string to validate

    Returns:
        Tuple of (is_valid, list_of_errors)

    Examples:
        >>> validate_urn("urn:nir:stato:legge:2020-12-30;178")
        (True, [])

        >>> validate_urn("invalid-urn")
        (False, ["Invalid URN format: ..."])
    """
    components = parse_urn(urn)
    return (components.is_valid, components.validation_errors)


def _validate_components(components: URNComponents) -> list:
    """
    Validate parsed URN components.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Validate tipo_atto
    if not components.tipo_atto:
        errors.append("Missing tipo_atto")
    elif components.tipo_atto not in VALID_ACT_TYPES:
        # Allow unknown types but log warning
        log.warning("Unknown act type in URN", tipo_atto=components.tipo_atto)

    # Validate date format
    if components.data:
        if not DATE_PATTERN.match(components.data):
            errors.append(f"Invalid date format: {components.data}")
        else:
            # Check date is valid
            try:
                year, month, day = map(int, components.data.split('-'))
                if not (1 <= month <= 12):
                    errors.append(f"Invalid month: {month}")
                if not (1 <= day <= 31):
                    errors.append(f"Invalid day: {day}")
            except ValueError:
                errors.append(f"Cannot parse date: {components.data}")

    # Note: Costituzione is special case - no number required
    # Other URNs with date but no number are allowed (partial references)

    # Validate allegato (must be positive integer)
    if components.allegato:
        try:
            allegato_num = int(components.allegato)
            if allegato_num < 1:
                errors.append(f"Allegato must be positive: {components.allegato}")
        except ValueError:
            errors.append(f"Invalid allegato format: {components.allegato}")

    # Validate estensione
    # Extensions can be: "bis", "ter", or with subpart: "octies.1"
    if components.estensione:
        ext_lower = components.estensione.lower()
        # Extract Latin part before any dot (e.g., "octies" from "octies.1")
        latin_part = ext_lower.split('.')[0] if '.' in ext_lower else ext_lower
        if latin_part and latin_part not in LATIN_EXTENSIONS:
            errors.append(f"Unknown article extension: {components.estensione}")

    # Validate data_versione
    if components.data_versione:
        if not DATE_PATTERN.match(components.data_versione):
            errors.append(f"Invalid version date format: {components.data_versione}")

    return errors


def canonicalize_urn(urn: str, include_url: bool = False) -> str:
    """
    Convert a URN to its canonical form.

    Normalizes the URN to a standard format:
    - Lowercase tipo_atto
    - Proper component ordering
    - Consistent formatting

    Args:
        urn: Input URN string
        include_url: If True, prepend Normattiva URL

    Returns:
        Canonicalized URN string

    Raises:
        ValueError: If URN is invalid
    """
    components = parse_urn(urn)

    if not components.is_valid:
        raise ValueError(f"Cannot canonicalize invalid URN: {components.validation_errors}")

    return components.to_urn(include_url=include_url)


def urns_are_equivalent(urn1: str, urn2: str, ignore_version: bool = True) -> bool:
    """
    Check if two URNs refer to the same legal norm.

    Args:
        urn1: First URN
        urn2: Second URN
        ignore_version: If True, ignores @originale and !vig= for comparison

    Returns:
        True if URNs refer to same norm
    """
    c1 = parse_urn(urn1)
    c2 = parse_urn(urn2)

    if not c1.is_valid or not c2.is_valid:
        return False

    if ignore_version:
        return c1.canonical_key() == c2.canonical_key()
    else:
        return c1.to_urn() == c2.to_urn()


def extract_article_info(urn: str) -> Optional[Dict[str, str]]:
    """
    Extract article information from a URN.

    Args:
        urn: URN string

    Returns:
        Dict with 'articolo' and 'estensione' keys, or None if no article
    """
    components = parse_urn(urn)

    if not components.articolo:
        return None

    return {
        'articolo': components.articolo,
        'estensione': components.estensione or '',
        'full_article': f"{components.articolo}{components.estensione or ''}",
    }


def urn_to_normattiva_url(urn: str) -> str:
    """
    Convert URN to full Normattiva URL.

    Args:
        urn: URN string (with or without urn:nir:stato: prefix)

    Returns:
        Full Normattiva URL
    """
    components = parse_urn(urn)
    return components.to_urn(include_url=True)


def dict_to_urn(data: Dict[str, Any], include_url: bool = False) -> str:
    """
    Build a URN from dictionary of components.

    Args:
        data: Dict with keys matching URNComponents fields
        include_url: If True, prepend Normattiva URL

    Returns:
        URN string
    """
    components = URNComponents(
        tipo_atto=data.get('tipo_atto', '').lower().replace(' ', '.'),
        data=data.get('data'),
        numero=data.get('numero'),
        allegato=data.get('allegato'),
        articolo=data.get('articolo'),
        estensione=data.get('estensione'),
        versione=data.get('versione'),
        data_versione=data.get('data_versione'),
    )
    return components.to_urn(include_url=include_url)


def normalize_act_type_for_urn(act_type: str) -> str:
    """
    Normalize act type string for URN format.

    Converts spaces to dots and lowercase.

    Args:
        act_type: Input act type (e.g., "decreto legge", "Regio Decreto")

    Returns:
        Normalized form (e.g., "decreto.legge", "regio.decreto")
    """
    return act_type.lower().strip().replace(' ', '.')


def extension_to_number(estensione: str) -> int:
    """
    Convert Latin ordinal extension to number.

    Args:
        estensione: Latin ordinal (e.g., "bis", "ter", "quater")

    Returns:
        Numeric equivalent (e.g., 2, 3, 4) or 0 if not found
    """
    return LATIN_EXTENSIONS.get(estensione.lower() if estensione else '', 0)


def number_to_extension(n: int) -> str:
    """
    Convert number to Latin ordinal extension.

    Args:
        n: Number (2-49)

    Returns:
        Latin ordinal (e.g., "bis" for 2) or empty string for 0/1
    """
    if n <= 1:
        return ''
    return _NUMBER_TO_EXTENSION.get(n, '')
