# Requirements: Josh Talks Hindi ASR Pipeline

**Defined:** 2026-03-26
**Core Value:** Domain-adapted Hindi ASR for Josh Talks conversational speech with lower WER than baseline, usable on free-tier GPU

## v1 Requirements

### Setup

- [ ] **SETUP-01**: All Python dependencies installed (`torch`, `torchaudio`, `transformers`, `datasets`, `evaluate`, `peft`, `bitsandbytes`, `fasttext-wheel`, `symspellpy`, `editdistance`, `jiwer`, `indic-nlp-library`, `numpy`, `pandas`, `requests`)
- [ ] **SETUP-02**: URL fixer function `fix_url()` implemented in `utils.py` (rewrites `goai_audio` → `upload_goai`)
- [ ] **SETUP-03**: Project directory structure created (`data/`, `hindi_dict/`, `models/`, `outputs/`)
- [ ] **SETUP-04**: Dataset metadata downloaded and all URLs rewritten via `fix_url()`

### Q1 — Whisper Fine-Tuning

- [ ] **Q1-01**: Audio loaded with `torchaudio`, resampled to 16kHz mono on-the-fly (no disk writes)
- [ ] **Q1-02**: Transcripts normalized with `indic-nlp-library` + Unicode NFC + Devanagari-only filter
- [ ] **Q1-03**: HuggingFace DatasetDict built with 90/10 train/test split (seed=42), clips 1–30s only
- [ ] **Q1-04**: Whisper-small loaded in 8-bit with `bitsandbytes` (`load_in_8bit=True`, `device_map="auto"`)
- [ ] **Q1-05**: LoRA adapter applied (`r=32`, `lora_alpha=64`, target `q_proj`+`v_proj`, `TaskType.SEQ_2_SEQ_LM`)
- [ ] **Q1-06**: `DataCollatorSpeechSeq2SeqWithPadding` implemented with label masking (-100)
- [ ] **Q1-07**: WER metric computed via `evaluate.load("wer")` during training
- [ ] **Q1-08**: Training run for 5 epochs, LoRA checkpoint saved to `./outputs/whisper-hi-lora/`
- [ ] **Q1-09**: Baseline (pretrained) vs. LoRA fine-tuned WER on FLEURS Hindi test set recorded in table
- [ ] **Q1-10**: 25 utterances sampled systematically (sorted by WER, every Nth, no cherry-picking)
- [ ] **Q1-11**: Error taxonomy documented (phonetic confusion, code-switching, numbers, proper nouns, disfluent speech)
- [ ] **Q1-12**: Top 3 fixes proposed with specific implementation steps
- [ ] **Q1-13**: At least 1 fix implemented with before/after WER on the 25 error subset

### Q2 — ASR Cleanup Pipeline

- [ ] **Q2-01**: Hindi number dictionaries implemented (`HINDI_UNITS`, `HINDI_TENS`, `HINDI_COMPOUND`, `HINDI_MULTIPLIERS`)
- [ ] **Q2-02**: Idiom guard patterns (`दो-चार`, `चार-पाँच`, etc.) freeze idioms before normalisation
- [ ] **Q2-03**: `normalize_numbers()` pipeline handles compound multipliers, standalone scales, simple words
- [ ] **Q2-04**: 4–5 before/after examples from actual data + 2–3 edge cases with reasoning
- [ ] **Q2-05**: FastText `.ftz` model downloaded (`lid.176.ftz`, <1 MB)
- [ ] **Q2-06**: Hindi Wordnet + loanword whitelist loaded as lookup sets
- [ ] **Q2-07**: `is_english_origin()` function applies three-step decision logic (Wordnet → whitelist → FastText)
- [ ] **Q2-08**: `tag_english_words()` wraps detected words in `[EN]...[/EN]` tags
- [ ] **Q2-09**: End-to-end pipeline demonstrated on actual ASR output examples

### Q3 — Spell Checker

- [ ] **Q3-01**: `build_hindi_frequency_dict()` built from Josh Talks transcripts + CC-100/Leipzig corpus
- [ ] **Q3-02**: Loanword Devanagari whitelist populated with common English loanwords in Devanagari
- [ ] **Q3-03**: SymSpell initialized with `max_dictionary_edit_distance=2`, Hindi frequency dictionary loaded
- [ ] **Q3-04**: Three-phase `classify_word()` pipeline: Wordnet whitelist → frequency signal → SymSpell edit distance
- [ ] **Q3-05**: All 1.77 lakh unique words classified with `label`, `confidence`, `reason`
- [ ] **Q3-06**: Final count: correctly spelled vs. incorrectly spelled unique words
- [ ] **Q3-07**: Output formatted for Google Sheets: `word | classification | confidence | reason`
- [ ] **Q3-08**: 40–50 low-confidence words manually reviewed with accuracy analysis
- [ ] **Q3-09**: 1–2 unreliable word categories identified (e.g. transliterated loanwords, proper nouns)

### Q4 — Lattice-WER

- [ ] **Q4-01**: Numeric variant mapping (`DIGIT_TO_HINDI`, `HINDI_TO_DIGIT`) implemented
- [ ] **Q4-02**: `align_hypothesis_to_reference()` using `SequenceMatcher` for word-level alignment
- [ ] **Q4-03**: `build_lattice()` constructs bins per position with consensus threshold (≥4 models agree)
- [ ] **Q4-04**: `get_numeric_variants()` automatically adds digit/word form to every bin
- [ ] **Q4-05**: `lattice_wer()` scores hypothesis against lattice (0 error if word in bin)
- [ ] **Q4-06**: `evaluate_all_models()` computes standard WER, Lattice-WER, delta, and verdict for all 5 models
- [ ] **Q4-07**: Results table produced: model | standard WER | lattice WER | delta | verdict
- [ ] **Q4-08**: Theory documented: bin system, consensus rule, why word-level is the right alignment unit

## v2 Requirements

### Future Improvements

- **V2-01**: KenLM n-gram re-scorer for phonetic confusion errors
- **V2-02**: Hinglish augmentation dataset integration (Hinglish Delite / HinglishPeople)
- **V2-03**: Custom vocabulary force-decoding for Josh Talks-specific proper nouns
- **V2-04**: Real-time inference API wrapping the full pipeline

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full Whisper fine-tuning (244M params) | OOM on free T4; LoRA gives comparable domain adaptation |
| BERT for English detection | 500× larger than FastText `.ftz` for a binary task |
| Neural spell checker | SymSpell edit distance is the correct, purpose-built algorithm |
| Real-time inference API | Batch research deliverable; API is out of scope for this assignment |
| Mobile / web deployment | Not required by the internship task |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SETUP-01 to SETUP-04 | Phase 1 | Pending |
| Q1-01 to Q1-13 | Phase 2 | Pending |
| Q2-01 to Q2-09 | Phase 3 | Pending |
| Q3-01 to Q3-09 | Phase 4 | Pending |
| Q4-01 to Q4-08 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 42 total
- Mapped to phases: 42
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-26*
*Last updated: 2026-03-26 after initialization*
