"""
q2_cleanup_pipeline.py — ASR post-processing cleanup for raw Whisper output.

Two-stage pipeline:
  Stage 1 — Number Normalisation:
    Converts Hindi number words to digits using regex + dictionaries.
    Guards idiomatic expressions (दो-चार, चार-पाँच, etc.) from conversion.

  Stage 2 — English Word Detection and Tagging:
    Uses FastText language ID model (lid.176.ftz, <1 MB) with Hindi Wordnet
    and a loanword whitelist to identify English-origin words and wrap them
    in [EN]...[/EN] tags for downstream processing.

Usage:
    from q2_cleanup_pipeline import cleanup_asr_output
    cleaned = cleanup_asr_output("उसने तीन सौ किताबें खरीदीं और interview दिया")
    # → "उसने 300 किताबें खरीदीं और [EN]interview[/EN] दिया"
"""

import re
import logging
from pathlib import Path
from typing import Optional

from utils import load_hindi_wordlist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path("models")
HINDI_DICT_DIR = Path("hindi_dict")

FASTTEXT_MODEL_PATH = MODELS_DIR / "lid.176.ftz"
HINDI_WORDNET_PATH = HINDI_DICT_DIR / "hindi_wordnet.txt"
LOANWORDS_PATH = HINDI_DICT_DIR / "loanwords_devanagari.txt"

# ---------------------------------------------------------------------------
# Stage 1 — Number Normalisation
# ---------------------------------------------------------------------------

# Core number dictionaries (Devanagari → integer)
HINDI_UNITS: dict[str, int] = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छह": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19,
}

HINDI_TENS: dict[str, int] = {
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}

HINDI_COMPOUND: dict[str, int] = {
    "पच्चीस": 25, "पैंतीस": 35, "पैंतालीस": 45, "पचपन": 55,
    "पैंसठ": 65, "पचहत्तर": 75, "पचासी": 85, "पचानवे": 95,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
}

HINDI_MULTIPLIERS: dict[str, int] = {
    "सौ": 100, "हज़ार": 1_000, "लाख": 100_000, "करोड़": 10_000_000,
}

# Combined lookup for standalone number words
ALL_NUMBER_WORDS: dict[str, int] = {**HINDI_UNITS, **HINDI_TENS, **HINDI_COMPOUND}

# Idiomatic expressions that must NOT be converted
# e.g. "दो-चार बातें" = "a few things" (not "2-4 things")
IDIOM_PATTERNS: list[str] = [
    r"दो-चार",     # "a few"
    r"चार-पाँच",   # "four or five" (idiomatic)
    r"तीन-चार",    # "three or four"
    r"सात-आठ",     # "seven or eight"
    r"पाँच-सात",   # "five or seven"
]

# Pre-compile the compound number regex once
_multiplier_keys = "|".join(re.escape(k) for k in HINDI_MULTIPLIERS.keys())
_unit_keys = "|".join(re.escape(k) for k in ALL_NUMBER_WORDS.keys())
_COMPOUND_PATTERN = re.compile(
    rf"({_unit_keys})\s+({_multiplier_keys})"
)
_SORTED_NUMBER_WORDS = sorted(ALL_NUMBER_WORDS.items(), key=lambda x: -len(x[0]))


def normalize_numbers(text: str) -> str:
    """
    Converts Hindi number words to digits, respecting idiom guards.

    Steps:
      1. Freeze idiomatic expressions with placeholder tokens
      2. Replace compound multiplier patterns (e.g. दस हज़ार → 10000)
      3. Replace standalone scale words (सौ → 100)
      4. Replace simple number words (longest-first to avoid partial matches)
      5. Restore frozen idioms

    Args:
        text: Raw Hindi text (ASR output or transcript).

    Returns:
        Text with number words converted to Arabic numerals.

    Examples:
        >>> normalize_numbers("उसने तीन सौ चौवन किताबें खरीदीं")
        'उसने 354 किताबें खरीदीं'
        >>> normalize_numbers("दो-चार बातें करनी हैं")
        'दो-चार बातें करनी हैं'
        >>> normalize_numbers("एक हज़ार रुपये दिए")
        '1000 रुपये दिए'
    """
    # Step 1: Freeze idioms with unique placeholders
    frozen: dict[str, str] = {}
    for i, pattern in enumerate(IDIOM_PATTERNS):
        placeholder = f"__IDIOM{i}__"
        match = re.search(pattern, text)
        if match:
            frozen[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder)

    # Step 2: Compound multiplier patterns (e.g. दस हज़ार → 10000)
    def replace_compound(m: re.Match) -> str:
        unit_val = ALL_NUMBER_WORDS.get(m.group(1), 1)
        mult_val = HINDI_MULTIPLIERS.get(m.group(2), 1)
        return str(unit_val * mult_val)

    text = _COMPOUND_PATTERN.sub(replace_compound, text)

    # Step 3: Standalone scale words (सौ → 100, हज़ार → 1000, etc.)
    for word, val in HINDI_MULTIPLIERS.items():
        text = re.sub(rf"\b{re.escape(word)}\b", str(val), text)

    # Step 4: Simple number words (longest first to prevent partial matches)
    for word, val in _SORTED_NUMBER_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\b", str(val), text)

    # Step 5: Restore frozen idioms
    for placeholder, original in frozen.items():
        text = text.replace(placeholder, original)

    return text


# ---------------------------------------------------------------------------
# Stage 2 — English Word Detection
# ---------------------------------------------------------------------------


class EnglishWordDetector:
    """
    Identifies English-origin words in Devanagari/mixed Hindi text.

    Decision logic per word:
      1. Found in Hindi Wordnet  → definitely Hindi → not English
      2. Found in loanword whitelist → accepted Devanagari form → not English
      3. FastText lang-ID predicts "en" → flag as English

    Args:
        fasttext_model_path: Path to the lid.176.ftz model file.
        hindi_wordnet_path: Path to Hindi Wordnet word list (one word/line).
        loanword_whitelist_path: Path to whitelisted Devanagari loanwords.
    """

    def __init__(
        self,
        fasttext_model_path: str = str(FASTTEXT_MODEL_PATH),
        hindi_wordnet_path: str = str(HINDI_WORDNET_PATH),
        loanword_whitelist_path: str = str(LOANWORDS_PATH),
    ) -> None:
        import fasttext  # noqa: PLC0415 — lazy import (not always installed)

        self._model = fasttext.load_model(fasttext_model_path)
        self._hindi_words = load_hindi_wordlist(hindi_wordnet_path)
        self._loan_whitelist = load_hindi_wordlist(loanword_whitelist_path)
        logger.info("EnglishWordDetector initialized.")

    def is_english_origin(self, word: str) -> bool:
        """Returns True if the word is of English origin."""
        # Definite Hindi → not English
        if word in self._hindi_words:
            return False
        # Known Devanagari loanword → not English
        if word in self._loan_whitelist:
            return False
        # FastText language ID
        predictions = self._model.predict(word, k=1)
        lang = predictions[0][0].replace("__label__", "")
        return lang == "en"

    def tag_english_words(self, text: str) -> str:
        """
        Wraps English-origin words in [EN]...[/EN] tags.

        Args:
            text: Hindi/Hinglish text to tag.

        Returns:
            Text with English-origin words tagged.

        Example:
            >>> detector.tag_english_words("मेरा इंटरव्यू बहुत अच्छा गया")
            'मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया'
        """
        words = text.split()
        tagged = [
            f"[EN]{w}[/EN]" if self.is_english_origin(w) else w
            for w in words
        ]
        return " ".join(tagged)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def cleanup_asr_output(
    raw_text: str,
    detector: Optional["EnglishWordDetector"] = None,
) -> str:
    """
    Full two-stage cleanup pipeline for raw ASR output.

    Stage 1: Number normalisation (regex + dictionary)
    Stage 2: English word tagging (FastText + Wordnet + whitelist)

    If no detector is provided, only Stage 1 runs (useful for testing
    without the FastText model downloaded).

    Args:
        raw_text: Raw output from the ASR model.
        detector: Optional pre-initialised EnglishWordDetector.

    Returns:
        Cleaned and tagged text.
    """
    text = normalize_numbers(raw_text)
    if detector is not None:
        text = detector.tag_english_words(text)
    return text


# ---------------------------------------------------------------------------
# Demo / Quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Stage 1: Number Normalisation Examples ===\n")

    EXAMPLES = [
        ("उसने तीन सौ चौवन किताबें खरीदीं", "Compound multiplier resolution"),
        ("दो-चार बातें करनी हैं", "Idiom guard — must NOT convert"),
        ("एक हज़ार रुपये दिए", "Multiplier pattern"),
        ("उसने पाँच लाख रुपये जीते", "Large number (lakh)"),
        ("बीस साल बाद वापस आए", "Simple tens"),
        ("तीन-चार घंटे लग गए", "Idiomatic 'three or four hours'"),
    ]

    print(f"{'Input':<45} {'Output':<30} {'Notes'}")
    print("-" * 100)
    for inp, note in EXAMPLES:
        out = normalize_numbers(inp)
        print(f"{inp:<45} {out:<30} {note}")

    print("\n=== Stage 2: English Word Tagging ===")
    print("(Requires lid.176.ftz model — run: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -P models/)")
    try:
        det = EnglishWordDetector()
        tag_examples = [
            "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
            "कंप्यूटर पर ऑनलाइन मीटिंग थी",
        ]
        for ex in tag_examples:
            print(f"  Input:  {ex}")
            print(f"  Output: {det.tag_english_words(ex)}\n")
    except Exception as exc:
        print(f"  Skipped (model not found): {exc}")
