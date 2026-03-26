# Roadmap — Josh Talks Hindi ASR Pipeline

**Milestone:** v1.0 — Complete 4-Question Internship Deliverable
**Created:** 2026-03-26
**Status:** 🔴 Not Started

---

## Phase 1 — Environment Setup & Project Foundation

**Goal:** Fully configured, runnable environment with scalable project structure and the critical URL fixer in place.

**Requirements:** SETUP-01, SETUP-02, SETUP-03, SETUP-04

**Success criteria:**
- `pip install` script runs cleanly in Colab/Kaggle T4
- `fix_url()` correctly transforms all 3 URL types (audio, transcription, metadata)
- All project directories created; `utils.py` is importable
- Metadata JSON downloaded and all URLs verified

**Plans:**
- 01-01: Create project directory structure and `utils.py` with URL fixer + shared helpers
- 01-02: Write environment setup scripts (`requirements.txt`, `setup.sh`, Colab setup cell)

---

## Phase 2 — Q1: Whisper Fine-Tuning Pipeline

**Goal:** Domain-adapted Whisper-small LoRA checkpoint on Josh Talks Hindi audio with documented WER improvement, error analysis, and at least 1 implemented fix.

**Requirements:** Q1-01 through Q1-13

**Success criteria:**
- `q1_finetune.py` runs end-to-end without errors on a T4 GPU
- LoRA checkpoint saved to `./outputs/whisper-hi-lora/` (~8 MB)
- WER table populated with baseline and fine-tuned numbers from FLEURS Hindi
- Error taxonomy covers at least 3 categories with examples
- 1 fix implemented with before/after WER delta on error subset

**Plans:**
- 02-01: Data preprocessing module (URL fix → torchaudio load → indic-nlp normalize → HF Dataset)
- 02-02: LoRA + 8-bit model setup, data collator, WER metric, and training arguments
- 02-03: Baseline evaluation on FLEURS Hindi, error sampling + taxonomy, fix implementation

---

## Phase 3 — Q2: ASR Cleanup Pipeline

**Goal:** Production-grade post-processor that converts raw Whisper output into clean Hindi text with numeric normalisation and English word tagging.

**Requirements:** Q2-01 through Q2-09

**Success criteria:**
- `q2_cleanup_pipeline.py` applies number normalisation + English tagging in a single pass
- All 6 idiom guard patterns tested and verified
- FastText `.ftz` model integrated with Wordnet + whitelist fallback
- 4–5 before/after examples from actual data documented

**Plans:**
- 03-01: Number normalisation pipeline (dictionaries, idiom guard, regex engine)
- 03-02: English word detector (FastText language ID + Wordnet + loanword whitelist + tagging)

---

## Phase 4 — Q3: Hindi Spell Checker

**Goal:** Fast, accurate spell checker that classifies all 1.77 lakh unique words in seconds using SymSpell with confidence scoring.

**Requirements:** Q3-01 through Q3-09

**Success criteria:**
- `q3_spellcheck.py` processes 177k words in <30 seconds
- Output CSV/TSV ready for Google Sheets import
- Low-confidence bucket reviewed with documented findings
- English loanwords in Devanagari correctly classified as "correct"

**Plans:**
- 04-01: Hindi frequency dictionary builder + loanword whitelist + SymSpell initialization
- 04-02: Three-phase classification pipeline, batch processor, output formatter, low-confidence review

---

## Phase 5 — Q4: Lattice-WER Evaluation

**Goal:** Fairer WER evaluation that does not penalise models for valid transcription alternatives (digit vs. word, spelling variants, synonyms).

**Requirements:** Q4-01 through Q4-08

**Success criteria:**
- `q4_lattice_wer.py` computes both standard WER and Lattice-WER for 5 models
- Results table shows model | std WER | lattice WER | delta | verdict
- Lattice construction logic is documented with theory + examples
- Consensus rule demonstrated with ≥4 models overriding reference

**Plans:**
- 05-01: Lattice construction (alignment, bin building, numeric variants, consensus rule)
- 05-02: Lattice-WER scorer, multi-model evaluator, results table, theory documentation

---

## Progress Summary

| Phase | Name | Status | Plans |
|-------|------|--------|-------|
| 1 | Environment Setup & Foundation | 🔴 Not Started | 2 plans |
| 2 | Q1: Whisper Fine-Tuning | 🔴 Not Started | 3 plans |
| 3 | Q2: ASR Cleanup Pipeline | 🔴 Not Started | 2 plans |
| 4 | Q3: Hindi Spell Checker | 🔴 Not Started | 2 plans |
| 5 | Q4: Lattice-WER Evaluation | 🔴 Not Started | 2 plans |

**Total:** 5 phases · 11 plans · 42 requirements

---
*Roadmap created: 2026-03-26*
