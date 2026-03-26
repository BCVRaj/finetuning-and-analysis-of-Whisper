"""
q3_spellcheck.py — Fast Hindi spell checker using SymSpell.

Classifies 1.77 lakh unique Hindi words as correctly/incorrectly spelled
in under 30 seconds using edit-distance + frequency signals.

Critical rule: English words transcribed in Devanagari (e.g. कंप्यूटर for
"computer") are CORRECT by Josh Talks transcription guidelines and must
never be flagged as errors.

Pipeline:
  Phase 1 — Whitelist check (Hindi Wordnet + loanword whitelist)
  Phase 2 — Corpus frequency signal (high freq = likely correct)
  Phase 3 — SymSpell edit distance lookup

Usage:
    python q3_spellcheck.py --input data/unique_words.txt --output outputs/spelling_results.csv
"""

import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from symspellpy import SymSpell, Verbosity

from utils import load_hindi_wordlist

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HINDI_DICT_DIR = Path("hindi_dict")
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")

HINDI_WORDNET_PATH = HINDI_DICT_DIR / "hindi_wordnet.txt"
HINDI_FREQUENCY_PATH = HINDI_DICT_DIR / "hindi_frequency.txt"
LOANWORDS_PATH = HINDI_DICT_DIR / "loanwords_devanagari.txt"

# ---------------------------------------------------------------------------
# Frequency Dictionary Builder
# ---------------------------------------------------------------------------

_DEVANAGARI_WORD_RE = re.compile(r"[\u0900-\u097F]+")


def build_hindi_frequency_dict(corpus_files: List[str], output_path: str) -> None:
    """
    Builds a SymSpell-compatible frequency dictionary from raw Hindi text corpora.

    Output format (one entry per line): "word frequency"

    Sources (recommended):
      - Josh Talks transcript text files
      - CC-100 Hindi corpus (https://data.statmt.org/cc-100/)
      - Leipzig Hindi corpus

    Args:
        corpus_files: List of paths to plain-text Hindi corpus files (UTF-8).
        output_path: Output path for the frequency dictionary.
    """
    word_counts: Counter[str] = Counter()

    for filepath in corpus_files:
        logger.info(f"Processing corpus: {filepath}")
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    words = _DEVANAGARI_WORD_RE.findall(line)
                    word_counts.update(words)
        except FileNotFoundError:
            logger.warning(f"Corpus file not found: {filepath}")

    logger.info(f"Writing {len(word_counts)} unique words to {output_path}")
    with open(output_path, "w", encoding="utf-8") as out:
        for word, count in word_counts.most_common():
            out.write(f"{word} {count}\n")


# ---------------------------------------------------------------------------
# SymSpell Initialisation
# ---------------------------------------------------------------------------


def init_symspell(frequency_dict_path: str) -> Tuple[SymSpell, Dict[str, int]]:
    """
    Initialises SymSpell with the Hindi frequency dictionary.

    Returns:
        Tuple of (SymSpell instance, frequency lookup dict)
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    loaded = sym_spell.load_dictionary(
        frequency_dict_path,
        term_index=0,
        count_index=1,
        encoding="utf-8",
    )
    if not loaded:
        raise FileNotFoundError(f"Could not load SymSpell dictionary from {frequency_dict_path}")

    # Build a plain frequency dict for fast lookup
    freq_dict: Dict[str, int] = {}
    try:
        with open(frequency_dict_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    freq_dict[parts[0]] = int(parts[1])
    except FileNotFoundError:
        pass

    logger.info(f"SymSpell loaded {len(freq_dict)} words")
    return sym_spell, freq_dict


# ---------------------------------------------------------------------------
# Three-Phase Classification Pipeline
# ---------------------------------------------------------------------------

HIGH_FREQ_THRESHOLD = 10   # ≥10 occurrences → high confidence correct
LOW_FREQ_THRESHOLD = 2     # 1–2 occurrences → suspicious


class HindiSpellChecker:
    """
    Three-phase Hindi spell checker.

    Phase 1: Whitelist (Hindi Wordnet + Devanagari loanwords) → always correct
    Phase 2: Corpus frequency signal → likely correct if high freq
    Phase 3: SymSpell edit distance → flags close-match misspellings
    """

    def __init__(
        self,
        symspell: SymSpell,
        freq_dict: Dict[str, int],
        hindi_wordnet: set,
        loan_whitelist: set,
    ) -> None:
        self._sym_spell = symspell
        self._freq_dict = freq_dict
        self._hindi_wordnet = hindi_wordnet
        self._loan_whitelist = loan_whitelist

    def classify(self, word: str) -> dict:
        """
        Classifies a single word.

        Returns:
            dict with keys: word, label, confidence, reason
            label: "correct" or "incorrect"
            confidence: "high", "medium", or "low"
        """
        freq = self._freq_dict.get(word, 0)

        # ── Phase 1: Hard whitelists ──────────────────────────────────────
        if word in self._hindi_wordnet:
            return self._result(word, "correct", "high", "Found in Hindi Wordnet")
        if word in self._loan_whitelist:
            return self._result(
                word, "correct", "high",
                "Known English loanword in Devanagari (per transcription guidelines)"
            )

        # ── Phase 2: Frequency signal ─────────────────────────────────────
        if freq >= HIGH_FREQ_THRESHOLD:
            return self._result(
                word, "correct", "medium",
                f"High corpus frequency ({freq} occurrences)"
            )

        # ── Phase 3: SymSpell edit distance ───────────────────────────────
        suggestions = self._sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

        if not suggestions:
            if freq >= LOW_FREQ_THRESHOLD:
                return self._result(
                    word, "correct", "low",
                    f"No spelling suggestion found; appears {freq} times in corpus"
                )
            return self._result(
                word, "incorrect", "medium",
                "No dictionary match and hapax legomenon (appears once or not at all)"
            )

        best = suggestions[0]

        if best.distance == 0:
            return self._result(
                word, "correct", "high",
                f"Exact match in frequency dictionary (corpus freq: {best.count})"
            )
        elif best.distance == 1:
            return self._result(
                word, "incorrect", "high",
                f"Edit distance 1 from '{best.term}' — likely typo or OCR error"
            )
        else:
            return self._result(
                word, "incorrect", "medium",
                f"Edit distance 2 from '{best.term}' — possible typo or very rare word"
            )

    @staticmethod
    def _result(word: str, label: str, confidence: str, reason: str) -> dict:
        return {"word": word, "label": label, "confidence": confidence, "reason": reason}

    def classify_batch(self, words: List[str]) -> List[dict]:
        """Classifies a list of words and returns results + summary statistics."""
        results = [self.classify(w) for w in words]
        correct = sum(1 for r in results if r["label"] == "correct")
        incorrect = len(results) - correct
        logger.info(
            f"Classified {len(results)} words: "
            f"{correct} correct ({correct/len(results)*100:.1f}%), "
            f"{incorrect} incorrect ({incorrect/len(results)*100:.1f}%)"
        )
        return results

    def get_low_confidence_bucket(self, results: List[dict]) -> List[dict]:
        """Returns all words classified with 'low' confidence for manual review."""
        return [r for r in results if r["confidence"] == "low"]


# ---------------------------------------------------------------------------
# Output Formatter
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "correct": "correct spelling",
    "incorrect": "incorrect spelling",
}


def save_results_csv(results: List[dict], output_path: str) -> None:
    """
    Saves classification results to a CSV file.

    Output columns: word | classification | confidence | reason
    Compatible with Google Sheets import.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "classification", "confidence", "reason"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "word": r["word"],
                "classification": _LABEL_MAP.get(r["label"], r["label"]),
                "confidence": r["confidence"],
                "reason": r["reason"],
            })
    logger.info(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Hindi spell checker for Josh Talks")
    parser.add_argument("--input", default="data/unique_words.txt",
                        help="Path to file with unique Hindi words (one per line)")
    parser.add_argument("--output", default="outputs/spelling_results.csv",
                        help="Path for output CSV")
    parser.add_argument("--build-freq-dict", action="store_true",
                        help="Build frequency dict from corpus first")
    args = parser.parse_args()

    # Optionally build frequency dict
    if args.build_freq_dict:
        corpus_files = [
            str(DATA_DIR / "josh_talks_transcripts.txt"),
            str(DATA_DIR / "cc100_hi_sample.txt"),
        ]
        build_hindi_frequency_dict(corpus_files, str(HINDI_FREQUENCY_PATH))

    # Initialise resources
    sym_spell, freq_dict = init_symspell(str(HINDI_FREQUENCY_PATH))
    hindi_wordnet = load_hindi_wordlist(str(HINDI_WORDNET_PATH))
    loan_whitelist = load_hindi_wordlist(str(LOANWORDS_PATH))

    checker = HindiSpellChecker(sym_spell, freq_dict, hindi_wordnet, loan_whitelist)

    # Load unique words to classify
    try:
        with open(args.input, encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Classifying {len(words)} unique words...")
    results = checker.classify_batch(words)

    # Save full results
    save_results_csv(results, args.output)

    # Low-confidence bucket report
    low_conf = checker.get_low_confidence_bucket(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"Low-confidence bucket: {len(low_conf)} words require manual review")
    logger.info("Known unreliable categories:")
    logger.info("  1. Devanagari-transliterated English loanwords (brand names, tech terms)")
    logger.info("  2. Proper nouns (names of people, cities, brands)")
    logger.info(f"{'='*60}")

    # Print summary statistics
    correct = sum(1 for r in results if r["label"] == "correct")
    print(f"\n=== Spell Check Summary ===")
    print(f"Total words classified : {len(results)}")
    print(f"Correct spellings      : {correct} ({correct/len(results)*100:.1f}%)")
    print(f"Incorrect spellings    : {len(results)-correct} ({(len(results)-correct)/len(results)*100:.1f}%)")
    print(f"Low confidence bucket  : {len(low_conf)} words")
    print(f"Results saved to       : {args.output}")


if __name__ == "__main__":
    main()
