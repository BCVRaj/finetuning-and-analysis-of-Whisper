"""
q4_lattice_wer.py — Lattice-WER evaluation for fairer multi-model ASR comparison.

Problem with standard WER:
  A model that outputs "14" for a reference that says "चौदह" is counted as wrong,
  even though both are valid transcriptions of the same audio. Lattice-WER
  fixes this by building a bin system where each position accepts all valid alternatives.

Algorithm:
  1. Align each model's output to the reference using SequenceMatcher
  2. Build a lattice: a list of bins, one per reference word position
  3. Each bin starts with the reference word and accumulates:
     - All model outputs at that position
     - Numeric variants (digit ↔ word form)
     - Consensus alternatives (if ≥N models agree on a different word, trust them)
  4. Score a hypothesis: 0 error if the output word is in the bin for that position

Usage:
    from q4_lattice_wer import evaluate_all_models

    results = evaluate_all_models(
        model_outputs=[model_a_tokens, model_b_tokens, ...],
        model_names=["Model A", "Model B", ...],
        reference_tokens=["उसने", "चौदह", "किताबें", "खरीदीं"],
    )
"""

import logging
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from jiwer import wer as standard_wer
# Note: editdistance is NOT used here — SequenceMatcher handles word alignment.
# Use editdistance only if you need character-level edit distance externally.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

# ---------------------------------------------------------------------------
# Real Data Path
# ---------------------------------------------------------------------------

Q4_DATA_PATH = Path("data") / "Question 4.xlsx"

# Model columns in the real Excel (in order)
MODEL_COLUMNS = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]

# ---------------------------------------------------------------------------
# Numeric Variant Mapping (digit ↔ Hindi word form)
# ---------------------------------------------------------------------------

DIGIT_TO_HINDI: Dict[str, str] = {
    "0": "शून्य",
    "1": "एक",
    "2": "दो",
    "3": "तीन",
    "4": "चार",
    "5": "पाँच",
    "6": "छह",
    "7": "सात",
    "8": "आठ",
    "9": "नौ",
    "10": "दस",
    "11": "ग्यारह",
    "12": "बारह",
    "13": "तेरह",
    "14": "चौदह",
    "15": "पंद्रह",
    "20": "बीस",
    "25": "पच्चीस",
    "30": "तीस",
    "50": "पचास",
    "100": "सौ",
    "1000": "हज़ार",
}

HINDI_TO_DIGIT: Dict[str, str] = {v: k for k, v in DIGIT_TO_HINDI.items()}


def get_numeric_variants(alternatives: Set[str]) -> Set[str]:
    """
    Expands a set of word alternatives with their numeric counterparts.

    E.g. if "चौदह" is in the set, adds "14"; if "14" is in the set, adds "चौदह".
    """
    extras: Set[str] = set()
    for alt in alternatives:
        if alt in DIGIT_TO_HINDI:
            extras.add(DIGIT_TO_HINDI[alt])
        if alt in HINDI_TO_DIGIT:
            extras.add(HINDI_TO_DIGIT[alt])
    return extras


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_hypothesis_to_reference(reference: List[str], hypothesis: List[str]) -> Dict[int, str]:
    """
    Aligns hypothesis words to reference positions using SequenceMatcher.

    Returns:
        Dict mapping reference_position → hypothesis_word at that position.
        Only positions with a matched hypothesis word are included.
    """
    matcher = SequenceMatcher(None, reference, hypothesis, autojunk=False)
    alignment: Dict[int, str] = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for ref_pos, hyp_pos in zip(range(i1, i2), range(j1, j2)):
                alignment[ref_pos] = hypothesis[hyp_pos]

    return alignment


# ---------------------------------------------------------------------------
# Lattice Construction
# ---------------------------------------------------------------------------


def build_lattice(
    model_outputs: List[List[str]],
    reference: List[str],
    consensus_threshold: int = 4,
) -> List[Set[str]]:
    """
    Constructs a word-level lattice from multiple model outputs and the human reference.

    A lattice is a list of bins — one per reference position. Each bin is a set of all
    valid transcription alternatives for that position.

    Consensus rule: if ≥ consensus_threshold models agree on a word different from the
    reference, trust the models and add their word to the bin. This handles cases where
    the written reference is wrong but all models heard the audio correctly.

    Args:
        model_outputs: List of tokenised hypothesis transcripts, one per model.
        reference: Tokenised human reference transcript.
        consensus_threshold: Minimum model agreement count to override the reference.

    Returns:
        List of sets — the lattice. lattice[i] is the set of valid words at position i.

    Example:
        reference = ["उसने", "चौदह", "किताबें", "खरीदीं"]
        After build_lattice with 3 models:
          lattice[0] = {"उसने"}
          lattice[1] = {"चौदह", "14"}           ← numeric variant added automatically
          lattice[2] = {"किताबें", "किताबे"}   ← model variant
          lattice[3] = {"खरीदीं", "खरीदी"}     ← gender variant
    """
    alignments = [
        align_hypothesis_to_reference(reference, hyp) for hyp in model_outputs
    ]
    lattice: List[Set[str]] = []

    for pos in range(len(reference)):
        bin_set: Set[str] = {reference[pos]}

        # Collect all model hypotheses at this position
        hyp_words_at_pos = [a[pos] for a in alignments if pos in a]
        bin_set.update(hyp_words_at_pos)

        # Consensus rule: if ≥ threshold models agree on a non-reference word, trust them
        if hyp_words_at_pos:
            top_word, top_count = Counter(hyp_words_at_pos).most_common(1)[0]
            if top_count >= consensus_threshold and top_word != reference[pos]:
                bin_set.add(top_word)
                logger.debug(
                    f"  Consensus override at pos {pos}: "
                    f"'{reference[pos]}' + '{top_word}' (agreed by {top_count} models)"
                )

        # Automatically add numeric variants (digit ↔ word form)
        bin_set |= get_numeric_variants(bin_set)
        lattice.append(bin_set)

    return lattice


# ---------------------------------------------------------------------------
# Lattice-WER Scoring
# ---------------------------------------------------------------------------


def lattice_wer(lattice: List[Set[str]], hypothesis: List[str]) -> float:
    """
    Computes Lattice-WER for a single hypothesis against a pre-built lattice.

    A word at position i is scored as CORRECT (0 error) if it appears in lattice[i].
    Insertions/deletions beyond the lattice length add full errors.

    Args:
        lattice: The lattice built by build_lattice().
        hypothesis: Tokenised hypothesis transcript.

    Returns:
        Lattice-WER as a float in [0, 1]. 0 = perfect, 1 = completely wrong.
    """
    if not lattice:
        return 1.0

    min_len = min(len(lattice), len(hypothesis))
    errors = sum(
        1 for pos in range(min_len)
        if hypothesis[pos] not in lattice[pos]
    )
    # Count length difference as deletions or insertions
    errors += abs(len(lattice) - len(hypothesis))

    return errors / len(lattice)


# ---------------------------------------------------------------------------
# Multi-Model Evaluation
# ---------------------------------------------------------------------------


def evaluate_all_models(
    model_outputs: List[List[str]],
    model_names: List[str],
    reference_tokens: List[str],
    consensus_threshold: int = 4,
) -> List[dict]:
    """
    Evaluates all models on a single reference with both standard WER and Lattice-WER.

    Args:
        model_outputs: List of tokenised hypotheses, one per model.
        model_names: Human-readable name for each model (same order as model_outputs).
        reference_tokens: Tokenised human reference.
        consensus_threshold: Passed to build_lattice().

    Returns:
        List of result dicts with keys:
          model, standard_wer, lattice_wer, delta, verdict
    """
    if len(model_outputs) != len(model_names):
        raise ValueError("model_outputs and model_names must have the same length")

    reference_str = " ".join(reference_tokens)
    lattice = build_lattice(model_outputs, reference_tokens, consensus_threshold)
    results: List[dict] = []

    for name, hyp_tokens in zip(model_names, model_outputs):
        std = standard_wer(reference_str, " ".join(hyp_tokens))
        lat = lattice_wer(lattice, hyp_tokens)
        delta = std - lat
        verdict = "Unfairly penalised" if delta > 0.01 else "Fairly scored"

        results.append({
            "model": name,
            "standard_wer": round(std, 4),
            "lattice_wer": round(lat, 4),
            "delta": round(delta, 4),
            "verdict": verdict,
        })

    return results


def print_results_table(results: List[dict]) -> None:
    """Pretty-prints the evaluation results table."""
    header = f"{'Model':<30} {'Std WER':>10} {'Lat WER':>10} {'Delta':>8} {'Verdict'}"
    print(f"\n{'='*90}")
    print(header)
    print(f"{'-'*90}")
    for r in results:
        print(
            f"{r['model']:<30} {r['standard_wer']:>10.4f} {r['lattice_wer']:>10.4f} "
            f"{r['delta']:>8.4f}  {r['verdict']}"
        )
    print(f"{'='*90}")

    # Explanation
    penalised = [r for r in results if r["verdict"] == "Unfairly penalised"]
    if penalised:
        print(
            f"\nNote: {len(penalised)} model(s) were unfairly penalised by standard WER for valid "
            "alternatives (e.g., outputting '14' when the reference says 'चौदह'). "
            "Lattice-WER corrects this."
        )
    else:
        print("\nAll models were fairly scored by standard WER for this utterance.")


# ---------------------------------------------------------------------------
# Real Data Loader
# ---------------------------------------------------------------------------


def load_q4_data(path: str = str(Q4_DATA_PATH)) -> Dict:
    """
    Loads Question 4.xlsx into structured form.

    Columns expected: segment_url_link, Human, Model H, Model i,
                      Model k, Model l, Model m, Model n

    Returns:
        Dict with keys:
          - segments: list of dicts with 'url', 'reference', 'models'
          - model_names: list of model name strings
    """
    df = pd.read_excel(path)
    # Drop fully empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    model_cols = [c for c in MODEL_COLUMNS if c in df.columns]
    logger.info(f"Loaded {len(df)} segments from {path}. Models: {model_cols}")

    segments = []
    for _, row in df.iterrows():
        ref = str(row.get("Human", "")).strip()
        if not ref or ref.lower() == "nan":
            continue
        model_outputs = {
            m: str(row.get(m, "")).strip() for m in model_cols
        }
        segments.append({
            "url": str(row.get("segment_url_link", "")).strip(),
            "reference": ref,
            "models": model_outputs,
        })
    return {"segments": segments, "model_names": model_cols}


def evaluate_from_excel(
    path: str = str(Q4_DATA_PATH),
    consensus_threshold: int = 4,
) -> None:
    """
    Full Lattice-WER evaluation on the real Question 4.xlsx data.

    For each of the 46 segments:
      - Tokenises the Human reference and all model outputs
      - Builds a lattice
      - Computes standard WER and Lattice-WER per model per segment

    Prints:
      - Per-model aggregate results table (averaged across all segments)
    """
    data = load_q4_data(path)
    segments = data["segments"]
    model_names = data["model_names"]

    # Accumulators: {model_name: [list of (std_wer, lat_wer)]}
    totals: Dict[str, List] = {m: [] for m in model_names}

    for seg_idx, seg in enumerate(segments):
        ref_tokens = seg["reference"].split()
        model_outputs = [seg["models"][m].split() for m in model_names]

        try:
            lattice = build_lattice(model_outputs, ref_tokens, consensus_threshold)
        except Exception as exc:
            logger.warning(f"Segment {seg_idx}: lattice build failed — {exc}")
            continue

        ref_str = seg["reference"]
        for m_name, hyp_tokens in zip(model_names, model_outputs):
            hyp_str = " ".join(hyp_tokens)
            try:
                std = standard_wer(ref_str, hyp_str)
            except Exception:
                std = 1.0
            lat = lattice_wer(lattice, hyp_tokens)
            totals[m_name].append((std, lat))

    # Aggregate and print results
    print(f"\n{'='*90}")
    print(f" Lattice-WER Evaluation — Question 4.xlsx ({len(segments)} segments)")
    print(f"{'='*90}")
    print(f"{'Model':<15} {'Avg Std WER':>12} {'Avg Lat WER':>12} {'Avg Delta':>10} {'Verdict'}")
    print(f"{'-'*90}")

    for m_name in model_names:
        scores = totals[m_name]
        if not scores:
            continue
        avg_std = sum(s for s, _ in scores) / len(scores)
        avg_lat = sum(l for _, l in scores) / len(scores)
        avg_delta = avg_std - avg_lat
        verdict = "Unfairly penalised" if avg_delta > 0.01 else "Fairly scored"
        print(
            f"{m_name:<15} {avg_std:>12.4f} {avg_lat:>12.4f} "
            f"{avg_delta:>10.4f}  {verdict}"
        )
    print(f"{'='*90}")
    print(
        "\nNote: 'Unfairly penalised' means standard WER over-penalised valid alternatives\n"
        "(e.g. digit vs word form). Lattice-WER corrects this."
    )


# ---------------------------------------------------------------------------
# Demo (hardcoded) — kept for quick unit testing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # ——— Run on real Question 4.xlsx data ———
    print("=" * 90)
    print(" REAL DATA: Lattice-WER evaluation on Question 4.xlsx")
    print("=" * 90)
    evaluate_from_excel()

    # ——— Hardcoded demo (sanity check) ———
    print("\n" + "="*90)
    print(" DEMO (hardcoded 5-model example for sanity check)")
    print("="*90)

    # Example utterance: "उसने चौदह किताबें खरीदीं"
    reference = ["उसने", "चौदह", "किताबें", "खरीदीं"]

    # Simulate 5 model outputs
    model_outputs = [
        ["उसने", "14", "किताबें", "खरीदीं"],       # Model A: digit form
        ["उसने", "चौदह", "किताबे", "खरीदी"],        # Model B: spelling + gender variant
        ["उसने", "14", "पुस्तकें", "खरीदीं"],       # Model C: digit + synonym
        ["उसने", "चौदह", "किताबें", "खरीदीं"],      # Model D: exact match
        ["उसने", "चौदह", "किताबें", "लिया"],         # Model E: wrong verb
    ]
    model_names = ["Model A", "Model B", "Model C", "Model D", "Model E"]

    print("=== Lattice-WER Evaluation Demo ===")
    print(f"\nReference: {' '.join(reference)}")
    print("\nBuilding lattice...")
    lattice = build_lattice(model_outputs, reference, consensus_threshold=3)
    for i, bin_set in enumerate(lattice):
        print(f"  Bin {i}: {bin_set}")

    results = evaluate_all_models(model_outputs, model_names, reference, consensus_threshold=3)
    print_results_table(results)
