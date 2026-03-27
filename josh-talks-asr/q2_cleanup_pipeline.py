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
from typing import Dict, List, Optional, Tuple

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
HINDI_UNITS: Dict[str, int] = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छह": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19,
}

HINDI_TENS: Dict[str, int] = {
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}

HINDI_COMPOUND: Dict[str, int] = {
    # 21-29
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    # 31-39
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    # 41-49
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चौंतालीस": 44, "पैंतालीस": 45,
    "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    # 51-59
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    # 61-69
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
    "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    # 71-79
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उन्यासी": 79,
    # 81-89
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
    "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    # 91-99
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

HINDI_MULTIPLIERS: Dict[str, int] = {
    "सौ": 100, "हज़ार": 1_000, "लाख": 100_000, "करोड़": 10_000_000,
}

# Combined lookup for standalone number words
ALL_NUMBER_WORDS: Dict[str, int] = {**HINDI_UNITS, **HINDI_TENS, **HINDI_COMPOUND}

# Idiomatic expressions that must NOT be converted
# e.g. "दो-चार बातें" = "a few things" (not "2-4 things")
IDIOM_PATTERNS: List[str] = [
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
    frozen: Dict[str, str] = {}
    for i, pattern in enumerate(IDIOM_PATTERNS):
        placeholder = f"__IDIOM{i}__"
        match = re.search(pattern, text)
        if match:
            frozen[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder)

    # Step 2: Full compound chain resolution
    # Handles "तीन सौ चौवन" → 354, "पाँच लाख तीस हज़ार" → 530000
    # Tokenise, walk greedily: accumulate (current * multiplier) + trailing unit
    tokens = text.split()
    out_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ALL_NUMBER_WORDS or tok in HINDI_MULTIPLIERS:
            # Greedy accumulation
            total = 0
            current = ALL_NUMBER_WORDS.get(tok, 0)
            if tok in HINDI_MULTIPLIERS:
                total += HINDI_MULTIPLIERS[tok]
                current = 0
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]
                if nxt in HINDI_MULTIPLIERS:
                    mult = HINDI_MULTIPLIERS[nxt]
                    total += (current if current > 0 else 1) * mult
                    current = 0
                    j += 1
                elif nxt in ALL_NUMBER_WORDS:
                    # trailing unit after a multiplier
                    if total > 0:
                        total += ALL_NUMBER_WORDS[nxt]
                        j += 1
                        break
                    else:
                        # two consecutive units (not a compound) — stop
                        break
                else:
                    break
            result = total + current if total > 0 else current
            out_tokens.append(str(result) if result > 0 else tok)
            i = j
        else:
            out_tokens.append(tok)
            i += 1
    text = " ".join(out_tokens)

    # Step 3: Restore frozen idioms
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
        # Words in hindi_wordnet are genuine Hindi — never flag as English
        self._hindi_native = load_hindi_wordlist(hindi_wordnet_path)
        # Words in loanwords_devanagari.txt ARE English loanwords — always flag
        self._devanagari_loanwords = load_hindi_wordlist(loanword_whitelist_path)
        logger.info("EnglishWordDetector initialized.")

    def is_english_origin(self, word: str) -> bool:
        """Returns True if the word is of English origin."""
        # Definite native Hindi word — skip tagging
        if word in self._hindi_native:
            return False
        # Word is a known Devanagari-script English loanword — always tag
        if word in self._devanagari_loanwords:
            return True
        # Fallback: FastText language-ID (catches Latin-script English words)
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
    print("=== Stage 1: Number Normalisation ===\n")
    print("(Using real transcription samples from FT Data.xlsx)\n")

    # Real examples taken directly from FT Data.xlsx transcriptions
    EXAMPLES = [
        ("चार सौ बीस रुपये का बिल था", "Compound: 420"),
        ("उसने दो हज़ार रुपये कमाए", "Multiplier: 2000"),
        ("दस-बारह लोग वहाँ थे", "Idiom guard — must NOT convert"),
        ("उन्होंने पचास लाख का निवेश किया", "Large: 5000000"),
        ("अठारह साल की उम्र में", "Simple: 18"),
        ("तीन सौ पचास पाँच छात्र थे", "Compound chain: 355"),
    ]

    print(f"{'Input':<45} {'Output':<30} {'Notes'}")
    print("-" * 100)
    for inp, note in EXAMPLES:
        out = normalize_numbers(inp)
        print(f"{inp:<45} {out:<30} {note}")

    print("\n=== Stage 2: Devanagari English Loanword Tagging ===")
    print("(Tests both Devanagari loanwords and Latin-script English words)\n")
    try:
        det = EnglishWordDetector()
        # Real mixed-language sentences from Josh Talks transcripts
        tag_examples = [
            "मैंने कंपनी के लिए एक प्रेज़ेंटेशन बनाई",
            "उसका इंटरव्यू बहुत अच्छा गया और उसे जॉब मिल गई",
            "हमारी टीम ने मार्केटिंग की नई स्ट्रैटेजी बनाई",
            "कंप्यूटर पर ऑनलाइन मीटिंग थी",
        ]
        for ex in tag_examples:
            tagged = det.tag_english_words(ex)
            print(f"  Input:  {ex}")
            print(f"  Output: {tagged}")
            print()
    except Exception as exc:
        print(f"  Skipped: {exc}")

