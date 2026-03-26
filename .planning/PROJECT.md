# Josh Talks — Hindi ASR Pipeline

## What This Is

A production-grade Hindi Automatic Speech Recognition (ASR) pipeline built for Josh Talks as an AI Researcher Intern assignment. The system fine-tunes Whisper-small using LoRA + 8-bit quantization on ~10 hours of Josh Talks Hindi audio, implements an ASR post-processing cleanup pipeline, builds a fast SymSpell spell-checker for 1.77 lakh Hindi words, and constructs a Lattice-WER evaluation framework for fair multi-model comparison.

## Core Value

Deliver accurate, domain-adapted Hindi ASR for Josh Talks conversational speech — achieving lower WER than the pretrained baseline — with a robust, efficient processing stack that runs on free-tier GPUs.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Phase 0 environment setup with all dependencies installed
- [ ] URL fixer implemented and all dataset URLs rewritten to `upload_goai` format
- [ ] Scalable project directory structure (`josh-talks-asr/`) in place
- [ ] Q1: Data preprocessed (torchaudio 16kHz, NFC normalized, 1-30s clips, 90/10 split)
- [ ] Q1: Whisper-small fine-tuned with LoRA (r=32) + 8-bit bitsandbytes on Josh Talks Hindi
- [ ] Q1: Baseline vs. fine-tuned WER comparison on FLEURS Hindi test set
- [ ] Q1: 25 systematically sampled error utterances with error taxonomy (3-5 categories)
- [ ] Q1: Top 3 error fixes proposed; at least 1 implemented with before/after WER
- [ ] Q2: Number normalisation pipeline (Hindi word → digit with idiom guard)
- [ ] Q2: English word detector using FastText `.ftz` (<1 MB) with `[EN]...[/EN]` tagging
- [ ] Q3: Hindi frequency dictionary built from Josh Talks transcripts + CC-100/Leipzig corpus
- [ ] Q3: SymSpell three-phase classifier on 1.77 lakh words with word | label | confidence | reason output
- [ ] Q3: Low-confidence bucket analysis (40–50 words manually reviewed)
- [ ] Q4: Lattice-WER system with bin-based reference lattice and consensus rule
- [ ] Q4: Standard WER vs. Lattice-WER delta computed for all 5 models

### Out of Scope

- Full fine-tuning (244M params) — LoRA covers domain adaptation without VRAM budget
- BERT-based English detection — FastText `.ftz` is purpose-built and 500× smaller
- BERT/neural spell checking — SymSpell edit distance is exact right tool for spell check
- Real-time inference pipeline — batch evaluation only
- Mobile app or API deployment — offline research deliverable

## Context

- **Dataset:** ~10 hours of Josh Talks Hindi conversational speech on Google Cloud Storage (metadata JSON with `rec_url_gcp`, `transcription_url`, `metadata_url`)
- **Critical:** All GCP URLs use legacy `goai_audio` bucket path; must be rewritten to `upload_goai` before any data access
- **GPU requirement:** Fine-tuning requires a T4 or better (Colab/Kaggle free tier sufficient with LoRA + 8-bit)
- **Language:** Hindi Devanagari with significant code-switching (Hinglish), proper nouns, and numeric utterances
- **Evaluation:** FLEURS Hindi test set as the standard benchmark for WER comparison

## Constraints

- **Hardware:** Must run on free Colab T4 (≤16 GB VRAM) — enforced by LoRA + 8-bit quantization
- **Model size:** LoRA checkpoint must be ~8 MB, not full 960 MB Whisper checkpoint
- **Speed:** Spell checker must process 177k words in seconds, not hours
- **Language library:** FastText `.ftz` model must be <1 MB (use `lid.176.ftz`)
- **No cherry-picking:** Error sampling must be systematic (sorted by severity, every Nth sample)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| LoRA r=32 over full fine-tune | Cuts trainable params from 244M → ~2M; runs on free T4 | — Pending |
| torchaudio over librosa | GPU-native, on-the-fly resampling, no disk duplication | — Pending |
| FastText `.ftz` over BERT for lang-ID | Binary task; `.ftz` is <1 MB vs 500 MB+, near-perfect accuracy | — Pending |
| SymSpell over neural spell check | Edit distance + frequency is the correct algorithm for this task | — Pending |
| Word-level lattice alignment | WER is word-level; bins are human-interpretable; cleaner semantics | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-26 after initialization*
