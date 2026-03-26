# STATE.md — Project Memory

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-26)

**Core value:** Domain-adapted Hindi ASR for Josh Talks conversational speech — lower WER than baseline, runs on free-tier GPU
**Current focus:** Phase 1 — Environment Setup & Project Foundation
**Milestone:** v1.0 — Complete 4-Question Internship Deliverable

---

## Current Position

**Phase:** 1 of 5 — Environment Setup & Project Foundation
**Status:** 🟡 Ready to execute

Next step: `/gsd-plan-phase 1`

---

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 1 | Environment Setup & Foundation | 🟡 Ready |
| 2 | Q1: Whisper Fine-Tuning | 🔴 Not Started |
| 3 | Q2: ASR Cleanup Pipeline | 🔴 Not Started |
| 4 | Q3: Hindi Spell Checker | 🔴 Not Started |
| 5 | Q4: Lattice-WER Evaluation | 🔴 Not Started |

---

## Project Structure Created

```
josh-talks-asr/
├── utils.py                      ✓ URL fixer + audio loader + transcript normalizer
├── q1_finetune.py                ✓ Whisper LoRA + 8-bit training pipeline
├── q2_cleanup_pipeline.py        ✓ Number normalisation + English tagger
├── q3_spellcheck.py              ✓ SymSpell three-phase classifier
├── q4_lattice_wer.py             ✓ Lattice-WER multi-model evaluator
├── requirements.txt              ✓ All dependency versions pinned
├── .gitignore                    ✓ Models/data excluded
├── data/                         → Download metadata.json here
├── hindi_dict/
│   └── loanwords_devanagari.txt  ✓ 50+ pre-populated loanwords
├── models/                       → Download lid.176.ftz here
└── outputs/
    └── whisper-hi-lora/          → LoRA checkpoint saved here after training
```

---

## Key Decisions Log

| Decision | Rationale | Phase |
|----------|-----------|-------|
| LoRA r=32 + 8-bit | Fits free T4, ~2M trainable params | Phase 1 |
| torchaudio (not librosa) | GPU-native, no disk duplication | Phase 1 |
| FastText `.ftz` | <1 MB, binary task, no BERT needed | Phase 3 |
| SymSpell | Milliseconds for 177k words vs hours for BERT | Phase 4 |
| Word-level lattice | WER is inherently word-level | Phase 5 |

---

## Session Notes

- Project initialized 2026-03-26
- Auto mode used: YOLO + Parallel + Balanced models + all agents enabled
- Source spec: project.md (861 lines, Josh Talks AI Researcher Intern task)

---
*STATE.md last updated: 2026-03-26 after project initialization*
