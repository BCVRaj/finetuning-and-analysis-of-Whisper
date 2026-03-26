# Josh Talks — AI Researcher Intern Task
## Speech & Audio | Project Plan & Implementation Guide (Optimized Stack)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Optimized Tech Stack & Why](#optimized-tech-stack--why)
3. [Phase 0 — Environment Setup](#phase-0--environment-setup)
4. [Question 1 — Whisper Fine-Tuning with LoRA + 8-bit](#question-1--whisper-fine-tuning-with-lora--8-bit)
5. [Question 2 — ASR Cleanup Pipeline](#question-2--asr-cleanup-pipeline)
6. [Question 3 — Spell Checker with SymSpell](#question-3--spell-checker-with-symspell)
7. [Question 4 — Lattice-WER Evaluation](#question-4--lattice-wer-evaluation)
8. [Recommended Timeline](#recommended-timeline)
9. [Deliverables Checklist](#deliverables-checklist)

---

## Project Overview

This project builds a complete Hindi ASR pipeline for Josh Talks across four components:

- Fine-tuning Whisper-small (with LoRA + 8-bit quantization) on ~10 hours of Hindi audio
- A post-processing cleanup pipeline for raw ASR output (number normalisation + English tagging)
- A fast, large-scale spell checker for 1.77 lakh Hindi words using SymSpell
- A Lattice-WER evaluation system for fair multi-model comparison

**Dataset:** ~10 hours of Hindi conversational speech with audio + transcription metadata on Google Cloud Storage.

**Critical — URL Format Fix:** All original dataset URLs must be rewritten to the `upload_goai` format before any data access. The fixer function is in Phase 0 — implement this first.

---

## Optimized Tech Stack & Why

This project deliberately avoids heavy neural models wherever they are not needed. Every tool below was chosen to be the right tool for the job, not just the most powerful one available.

| Task | Old (Heavy) Approach | Optimized Approach | Why Optimized Wins |
|---|---|---|---|
| Fine-tuning | Full fine-tune (244M params) | LoRA + 8-bit (`peft` + `bitsandbytes`) | Updates only ~2M params, cuts VRAM by 75%, runs on free Colab T4 |
| Audio loading | `librosa` (pure Python, loads all to RAM) | `torchaudio` + HF streaming | GPU-native, processes on-the-fly, no disk duplication |
| English detection | BERT (500 MB+) | FastText `.ftz` compressed model (<1 MB) | Binary lang-ID task — semantics not needed, `.ftz` is near-perfect |
| Spell checking | BERT embeddings (hours on 177k words) | `symspellpy` (milliseconds) | Edit distance + frequency is exactly what spell checking needs |
| WER / alignment | Heavy NLP pipelines | `editdistance` + `jiwer` | Pure algorithmic — no neural network needed here at all |

---

## Phase 0 — Environment Setup

### Installation

```bash
pip install torch torchaudio transformers datasets evaluate \
            peft bitsandbytes fasttext-wheel symspellpy \
            editdistance jiwer indic-nlp-library \
            numpy pandas requests
```

> **GPU note:** Fine-tuning Whisper-small requires a GPU. With LoRA + 8-bit you can run comfortably on a free Kaggle T4 or Google Colab T4. Training time: ~1.5–2 hrs (vs 3–6 hrs without LoRA).

### URL Fixer — Write This First

All data access depends on this. Every URL in the dataset metadata must go through this function before use.

```python
def fix_url(old_url: str) -> str:
    """
    Rewrites legacy GCP URLs to the upload_goai format.

    Example:
      Input:  https://storage.googleapis.com/goai_audio/967179/825780.wav
      Output: https://storage.googleapis.com/upload_goai/967179/825780.wav

    Transcription URL pattern:
      https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
    """
    return old_url.replace(
        "storage.googleapis.com/goai_audio",
        "storage.googleapis.com/upload_goai"
    )
```

### Project Folder Structure

```
josh-talks-asr/
├── data/
│   └── metadata.json             # Downloaded dataset metadata
├── hindi_dict/
│   ├── hindi_wordnet.txt         # Hindi Wordnet word list
│   ├── hindi_frequency.txt       # Hindi frequency corpus (CC-100 or Leipzig)
│   └── loanwords_devanagari.txt  # Whitelisted English words in Devanagari
├── models/
│   └── lid.176.ftz               # FastText compressed language ID model (<1 MB)
├── outputs/
│   └── whisper-hi-lora/          # LoRA adapter checkpoints
├── q1_finetune.py
├── q2_cleanup_pipeline.py
├── q3_spellcheck.py
├── q4_lattice_wer.py
└── utils.py                      # URL fixer + shared helpers
```

---

## Question 1 — Whisper Fine-Tuning with LoRA + 8-bit

**Goal:** Domain-adapt Whisper-small to Josh Talks' Hindi conversational audio without full fine-tuning overhead.

---

### Step a — Data Preprocessing

#### 1. Download Metadata & Fix URLs

```python
import json
import requests

def load_transcription(transcription_url: str) -> str:
    response = requests.get(fix_url(transcription_url))
    return response.json().get("transcription", "")

def load_metadata(path: str) -> list:
    with open(path) as f:
        records = json.load(f)
    # Fix all URLs in place
    for r in records:
        r["rec_url_gcp"]       = fix_url(r["rec_url_gcp"])
        r["transcription_url"] = fix_url(r["transcription_url"])
        r["metadata_url"]      = fix_url(r["metadata_url"])
    return records
```

#### 2. Audio Loading with torchaudio (On-the-Fly, No Disk Duplication)

`torchaudio` resamples directly in memory during the data-loading step — no need to save resampled `.wav` files to disk.

```python
import torchaudio
import torch
import io

TARGET_SR = 16000

def load_audio_torchaudio(url: str) -> torch.Tensor:
    """
    Streams audio from GCP URL, resamples to 16kHz on-the-fly.
    Returns a 1D float32 numpy array ready for Whisper's feature extractor.
    """
    response    = requests.get(url, stream=True)
    audio_bytes = io.BytesIO(response.content)
    waveform, orig_sr = torchaudio.load(audio_bytes)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
        waveform  = resampler(waveform)

    return waveform.squeeze().numpy()   # Shape: (num_samples,)
```

#### 3. Text Normalisation

```python
import re
import unicodedata
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

normalizer = IndicNormalizerFactory().get_normalizer("hi")

def normalize_transcript(text: str) -> str:
    text = normalizer.normalize(text)                   # Devanagari script normalisation
    text = unicodedata.normalize("NFC", text)           # Unicode composition
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)     # Keep only Devanagari + spaces
    text = re.sub(r"\s+", " ", text).strip()            # Collapse whitespace
    return text
```

#### 4. Build HuggingFace Dataset

```python
from datasets import Dataset, DatasetDict
from transformers import WhisperProcessor

MODEL_ID  = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="hi", task="transcribe")

def prepare_record(item: dict) -> dict:
    audio_array    = load_audio_torchaudio(item["rec_url_gcp"])
    transcript     = normalize_transcript(load_transcription(item["transcription_url"]))
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=TARGET_SR
    ).input_features[0]
    labels = processor.tokenizer(transcript).input_ids
    return {"input_features": input_features, "labels": labels, "duration": item["duration"]}

def build_dataset(metadata: list) -> DatasetDict:
    metadata = [m for m in metadata if 1.0 <= m["duration"] <= 30.0]
    records  = [prepare_record(m) for m in metadata]
    dataset  = Dataset.from_list(records)
    return dataset.train_test_split(test_size=0.1, seed=42)
```

**Preprocessing checklist:**

- [ ] All URLs rewritten via `fix_url()`
- [ ] Audio resampled to 16kHz mono via `torchaudio` (no saved files)
- [ ] Transcripts Unicode-NFC normalised
- [ ] Non-Devanagari characters stripped
- [ ] Clips outside 1–30 second range removed
- [ ] 90/10 train/validation split with fixed seed

---

### Step b — LoRA + 8-bit Fine-Tuning

#### Why LoRA over Full Fine-Tuning

| Metric | Full Fine-Tune | LoRA (r=32) |
|---|---|---|
| Trainable parameters | 244 million | ~2 million |
| VRAM required | ~10 GB | ~4 GB |
| Checkpoint size | ~960 MB | ~8 MB |
| Accuracy difference | Baseline | Negligible for domain adaptation |
| Runs on free Colab T4 | No (OOM risk) | Yes |

#### Load Model in 8-bit

```python
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    load_in_8bit=True,      # bitsandbytes — cuts VRAM by ~75%
    device_map="auto",
)
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []
```

#### Apply LoRA Adapter

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32,                                       # Rank
    lora_alpha=64,                              # Scaling (usually 2 × r)
    target_modules=["q_proj", "v_proj"],        # Attention projection layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params: ~2M || all params: 244M || trainable%: ~0.82%
```

#### Data Collator

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch          = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels         = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
```

#### WER Metric

```python
import evaluate

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}
```

#### Training Arguments

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs/whisper-hi-lora",
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,      # Effective batch size = 16
    warmup_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    fp16=True,
    report_to="tensorboard",
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
model.save_pretrained("./outputs/whisper-hi-lora")   # Saves only ~8 MB LoRA weights
```

---

### Step c — Baseline vs. Fine-Tuned WER on FLEURS Hindi

```python
from datasets import load_dataset
fleurs_hi_test = load_dataset("google/fleurs", "hi_in", split="test")
```

| Model | Dataset | WER (%) |
|---|---|---|
| Whisper-small — pretrained baseline | FLEURS Hindi test | — |
| Whisper-small — LoRA fine-tuned on Josh Talks | FLEURS Hindi test | — |

---

### Step d — Systematic Error Sampling (25 Utterances)

**Strategy — sorted by severity, every Nth sample (no cherry-picking):**

```python
def sample_errors_systematically(predictions: list, references: list, n: int = 25) -> list:
    per_utt = [
        {"idx": i, "pred": p, "ref": r,
         "wer": wer_metric.compute(predictions=[p], references=[r])}
        for i, (p, r) in enumerate(zip(predictions, references))
        if p.strip() != r.strip()
    ]
    per_utt.sort(key=lambda x: x["wer"], reverse=True)
    step    = max(1, len(per_utt) // n)
    sampled = per_utt[::step][:n]
    return sampled
```

---

### Step e — Error Taxonomy

Categories should emerge from the data. Likely categories for Josh Talks Hindi:

| Category | Description | Root Cause |
|---|---|---|
| Code-switching | Hinglish words misrecognised | Insufficient English-in-Hindi training data |
| Phonetic confusion | Similar-sounding word substitutions | Acoustic ambiguity in conversational speech |
| Number / date errors | Numbers wrong or omitted | Low numeric example frequency in training |
| Proper nouns | Names of speakers, cities, brands wrong | Out-of-vocabulary tokens |
| Fast / disfluent speech | Words merged, dropped, or repeated | High speech rate, filler words |

For each category document: reference transcript → model output → reasoning about the cause.

---

### Step f+g — Top 3 Fixes (Propose + Implement One)

**Fix 1 — Code-switching errors:** Augment training data with Hinglish examples (English words in Devanagari). Source from the Hinglish Delite or HinglishPeople datasets. Add 500–1000 such examples before re-running LoRA.

**Fix 2 — Phonetic confusion:** Post-processing re-scorer using a lightweight Hindi n-gram language model (KenLM). Score the top-2 Whisper beam search candidates and pick the higher language model score.

**Fix 3 — Proper noun errors:** Build a custom vocabulary list of Josh Talks-specific entities (speaker names, topic keywords, city names). Force-decode these tokens via `processor.tokenizer` before inference.

Show before/after WER on the 25 sampled utterances for the fix you implement.

---

## Question 2 — ASR Cleanup Pipeline

**Goal:** Post-process raw Whisper-small output (run the *pretrained* model, before fine-tuning) to make it usable for downstream tasks.

**Setup:** Run pretrained Whisper-small on all audio segments first. Pair each raw ASR output with the human reference from the JSON files.

---

### Part a — Number Normalisation (Pure Regex — 0 MB overhead)

No neural model needed. A deterministic regex + dictionary pipeline handles all cases including the idiom guard.

#### Number Dictionaries

```python
HINDI_UNITS = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छह": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19,
}
HINDI_TENS = {
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}
HINDI_COMPOUND = {
    "पच्चीस": 25, "पैंतीस": 35, "पैंतालीस": 45, "पचपन": 55,
    "पैंसठ": 65, "पचहत्तर": 75, "पचासी": 85, "पचानवे": 95,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
}
HINDI_MULTIPLIERS = {
    "सौ": 100, "हज़ार": 1000, "लाख": 100_000, "करोड़": 10_000_000,
}
ALL_NUMBER_WORDS = {**HINDI_UNITS, **HINDI_TENS, **HINDI_COMPOUND}
```

#### Idiom Guard + Normaliser

```python
import re

# Number words used idiomatically — must NOT be converted
IDIOM_PATTERNS = [
    r"दो-चार",    # "a few"       → stays as दो-चार, not "2-4"
    r"चार-पाँच",  # "four or five" → idiomatic quantity
    r"तीन-चार",   # "three or four"
    r"सात-आठ",    # "seven or eight"
    r"पाँच-सात",  # "five or seven"
]

def normalize_numbers(text: str) -> str:
    # Step 1: Freeze idioms with placeholders
    frozen = {}
    for i, pattern in enumerate(IDIOM_PATTERNS):
        placeholder = f"__IDIOM{i}__"
        match = re.search(pattern, text)
        if match:
            frozen[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder)

    # Step 2: Compound multiplier patterns (e.g. दस हज़ार → 10000)
    def replace_compound(m):
        unit_val = ALL_NUMBER_WORDS.get(m.group(1), 1)
        mult_val = HINDI_MULTIPLIERS.get(m.group(2), 1)
        return str(unit_val * mult_val)

    mult_pattern = "(" + "|".join(ALL_NUMBER_WORDS.keys()) + r")\s+(" \
                   + "|".join(HINDI_MULTIPLIERS.keys()) + ")"
    text = re.sub(mult_pattern, replace_compound, text)

    # Step 3: Standalone scale words (सौ → 100)
    for word, val in HINDI_MULTIPLIERS.items():
        text = re.sub(rf"\b{word}\b", str(val), text)

    # Step 4: Simple number words (longest first to avoid partial matches)
    for word, val in sorted(ALL_NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{word}\b", str(val), text)

    # Step 5: Restore frozen idioms
    for placeholder, original in frozen.items():
        text = text.replace(placeholder, original)

    return text
```

**Required deliverable:** 4–5 before/after examples from actual data + 2–3 edge case examples with reasoning.

| Input (raw ASR) | Output (after pipeline) | Notes |
|---|---|---|
| उसने तीन सौ चौवन किताबें खरीदीं | उसने 354 किताबें खरीदीं | Compound correctly resolved |
| दो-चार बातें करनी हैं | दो-चार बातें करनी हैं | Idiom guard preserved correctly |
| एक हज़ार रुपये दिए | 1000 रुपये दिए | Multiplier pattern matched |

---

### Part b — English Word Detection (FastText `.ftz` — <1 MB)

**Why not BERT:** This is binary classification at the word level — is this word of English or Hindi origin? FastText's language ID model is purpose-built for exactly this task and is ~500× smaller than BERT.

#### Download the Model

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -P models/
```

#### Implementation

```python
import fasttext

LID_MODEL      = fasttext.load_model("models/lid.176.ftz")
HINDI_WORDS    = load_hindi_wordlist("hindi_dict/hindi_wordnet.txt")
LOAN_WHITELIST = load_hindi_wordlist("hindi_dict/loanwords_devanagari.txt")

def is_english_origin(word: str) -> bool:
    """
    Decision logic:
      1. In Hindi Wordnet        → definitely Hindi  → False
      2. In loanword whitelist   → correct Devanagari → False
      3. FastText LID predicts English               → True
    """
    if word in HINDI_WORDS:
        return False
    if word in LOAN_WHITELIST:
        return False
    predictions = LID_MODEL.predict(word, k=1)
    lang = predictions[0][0].replace("__label__", "")
    return lang == "en"

def tag_english_words(text: str) -> str:
    """
    Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
    Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"
    """
    words  = text.split()
    tagged = [f"[EN]{w}[/EN]" if is_english_origin(w) else w for w in words]
    return " ".join(tagged)
```

> **Loanword whitelist tip:** Pre-populate `loanwords_devanagari.txt` with common Hindi-English words: इंटरव्यू, जॉब, कंप्यूटर, मोबाइल, ऑफिस, टीम, मैनेजर, वीडियो, ऑनलाइन, etc.

---

## Question 3 — Spell Checker with SymSpell

**Goal:** Classify all 1.77 lakh unique words as correctly or incorrectly spelled, with a confidence score. Fast — runs in seconds, not hours.

**Critical rule:** English words transcribed in Devanagari (e.g. `कंप्यूटर` for "computer") are **correct** by Josh Talks' transcription guidelines. They must never be flagged as errors.

### Why SymSpell over BERT

| Approach | Time on 177k words | Memory | Right tool? |
|---|---|---|---|
| BERT embedding similarity | ~4–8 hours | 1–4 GB GPU | No — overkill for spell check |
| SymSpell (edit distance + freq) | < 5 seconds | ~30 MB RAM | Yes — purpose-built for this |

---

### Building a Hindi Frequency Dictionary

SymSpell's default dictionary is English. You need a Hindi one.

```python
from collections import Counter
import re

def build_hindi_frequency_dict(corpus_files: list, output_path: str):
    """
    Builds a SymSpell-compatible frequency dictionary from raw Hindi text.
    Output format: "word frequency" one per line.
    Sources: Josh Talks transcripts + CC-100 Hindi or Leipzig Hindi corpus.
    """
    word_counts = Counter()
    for filepath in corpus_files:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                words = re.findall(r"[\u0900-\u097F]+", line)
                word_counts.update(words)

    with open(output_path, "w", encoding="utf-8") as out:
        for word, count in word_counts.most_common():
            out.write(f"{word} {count}\n")

build_hindi_frequency_dict(
    corpus_files=["data/josh_talks_transcripts.txt", "data/cc100_hi_sample.txt"],
    output_path="hindi_dict/hindi_frequency.txt"
)
```

---

### Three-Phase Classification Pipeline

```python
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(
    "hindi_dict/hindi_frequency.txt",
    term_index=0, count_index=1, encoding="utf-8"
)

HIGH_FREQ_THRESHOLD = 10   # 10+ occurrences → high confidence correct
LOW_FREQ_THRESHOLD  = 2    # 1–2 occurrences → suspicious

def classify_word(word: str, freq_dict: dict) -> dict:
    freq = freq_dict.get(word, 0)

    # Phase 1: Hard whitelist (always correct, always high confidence)
    if word in HINDI_WORDNET:
        return {"label": "correct", "confidence": "high",
                "reason": "Found in Hindi Wordnet"}
    if word in LOAN_WHITELIST:
        return {"label": "correct", "confidence": "high",
                "reason": "Known English loanword in Devanagari (per transcription guidelines)"}

    # Phase 2: Frequency signal
    if freq >= HIGH_FREQ_THRESHOLD:
        return {"label": "correct", "confidence": "medium",
                "reason": f"High corpus frequency ({freq} occurrences)"}

    # Phase 3: SymSpell edit distance check
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

    if not suggestions:
        if freq >= LOW_FREQ_THRESHOLD:
            return {"label": "correct", "confidence": "low",
                    "reason": f"No spelling suggestion; appears {freq} times"}
        return {"label": "incorrect", "confidence": "medium",
                "reason": "No dictionary match and hapax legomenon (appears once)"}

    best = suggestions[0]

    if best.distance == 0:
        return {"label": "correct", "confidence": "high",
                "reason": f"Exact match in frequency dictionary (freq: {best.count})"}
    elif best.distance == 1:
        return {"label": "incorrect", "confidence": "high",
                "reason": f"Edit distance 1 from '{best.term}' — likely typo"}
    else:
        return {"label": "incorrect", "confidence": "medium",
                "reason": f"Edit distance 2 from '{best.term}' — possible typo or rare word"}
```

---

### Output Format (Google Sheet)

| word | classification | confidence | reason |
|---|---|---|---|
| कंप्यूटर | correct spelling | high | Known English loanword in Devanagari |
| कंप्यटूर | incorrect spelling | high | Edit distance 1 from कंप्यूटर — likely typo |
| मुस्कुराहट | correct spelling | high | Exact match in frequency dictionary |
| मुस्करहट | incorrect spelling | high | Edit distance 1 from मुस्कुराहट |

---

### Low Confidence Bucket Review (Step c)

Manually review 40–50 words from the `low` confidence bucket. Document:

- Your human judgment: correct or incorrect?
- Whether the system agreed or disagreed
- Why the system got it wrong (if it did)

**Known unreliable categories:**

1. **Devanagari-transliterated English loanwords** — New or rare ones (brand names, tech terms) won't appear in the frequency dictionary and may get false "incorrect" labels. Fix: expand the loanword whitelist iteratively.
2. **Proper nouns** — Names of people, cities, and brands always miss the dictionary even when correctly spelled. They score low frequency and no dictionary match, leading to false positives.

---

## Question 4 — Lattice-WER Evaluation

**Goal:** Build a fairer WER that does not penalise models for valid transcription alternatives (digit vs. word, spelling variants, synonyms).

**Alignment unit:** Word-level. Reasons: WER is inherently word-level, bins are human-interpretable, and insertion/deletion semantics are cleaner at word boundaries.

**Tools:** `editdistance` (C-optimized) + `jiwer`. No neural network needed — this is pure algorithmic alignment.

---

### Core Concept: The Bin System

Instead of comparing against a single rigid reference string, build a **lattice** — a sequential list of bins. Each bin contains all valid alternatives for that position.

```
Audio:  "उसने चौदह किताबें खरीदीं"

Rigid:  ["उसने", "चौदह", "किताबें", "खरीदीं"]

Lattice:
  Bin 0: ["उसने"]
  Bin 1: ["चौदह", "14"]                   ← digit and word both valid
  Bin 2: ["किताबें", "किताबे", "पुस्तकें"]  ← spelling variant + synonym
  Bin 3: ["खरीदीं", "खरीदी"]               ← gender variant
```

A model output word scores 0 error if it appears anywhere in its corresponding bin.

---

### Implementation

```python
import editdistance
from collections import Counter
from difflib import SequenceMatcher
from jiwer import wer as standard_wer

# Numeric variant mapping
DIGIT_TO_HINDI = {
    "1":"एक", "2":"दो", "3":"तीन", "4":"चार", "5":"पाँच",
    "10":"दस", "12":"बारह", "14":"चौदह", "20":"बीस",
    "100":"सौ", "1000":"हज़ार",
}
HINDI_TO_DIGIT = {v: k for k, v in DIGIT_TO_HINDI.items()}

def get_numeric_variants(alternatives: set) -> set:
    extras = set()
    for alt in alternatives:
        if alt in DIGIT_TO_HINDI: extras.add(DIGIT_TO_HINDI[alt])
        if alt in HINDI_TO_DIGIT: extras.add(HINDI_TO_DIGIT[alt])
    return extras

def align_hypothesis_to_reference(reference: list, hypothesis: list) -> dict:
    matcher   = SequenceMatcher(None, reference, hypothesis)
    alignment = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for ref_pos, hyp_pos in zip(range(i1, i2), range(j1, j2)):
                alignment[ref_pos] = hypothesis[hyp_pos]
    return alignment

def build_lattice(model_outputs: list, reference: list,
                  consensus_threshold: int = 4) -> list:
    """
    Constructs a word-level lattice from multiple model outputs + a human reference.

    consensus_threshold: if this many models agree on a word different from the
    reference, trust the models and add their word to the bin.
    """
    alignments = [align_hypothesis_to_reference(reference, hyp) for hyp in model_outputs]
    lattice    = []

    for pos in range(len(reference)):
        bin_set          = {reference[pos]}
        hyp_words_at_pos = [a[pos] for a in alignments if pos in a]
        bin_set.update(hyp_words_at_pos)

        if hyp_words_at_pos:
            top_word, top_count = Counter(hyp_words_at_pos).most_common(1)[0]
            if top_count >= consensus_threshold and top_word != reference[pos]:
                bin_set.add(top_word)   # Models know better than the reference

        bin_set |= get_numeric_variants(bin_set)
        lattice.append(bin_set)

    return lattice

def lattice_wer(lattice: list, hypothesis: list) -> float:
    if not lattice:
        return 1.0
    min_len = min(len(lattice), len(hypothesis))
    errors  = sum(1 for pos in range(min_len) if hypothesis[pos] not in lattice[pos])
    errors += abs(len(lattice) - len(hypothesis))   # deletions + insertions
    return errors / len(lattice)

def evaluate_all_models(model_outputs: list, model_names: list,
                        reference_tokens: list) -> list:
    reference_str = " ".join(reference_tokens)
    lattice       = build_lattice(model_outputs, reference_tokens)
    results       = []

    for name, hyp_tokens in zip(model_names, model_outputs):
        std   = standard_wer(reference_str, " ".join(hyp_tokens))
        lat   = lattice_wer(lattice, hyp_tokens)
        delta = std - lat
        results.append({
            "model":        name,
            "standard_wer": round(std,   4),
            "lattice_wer":  round(lat,   4),
            "delta":        round(delta, 4),
            "verdict":      "Unfairly penalised" if delta > 0.01 else "Fairly scored",
        })

    return results
```

---

### Expected Output Table

| Model | Standard WER | Lattice-WER | Delta | Verdict |
|---|---|---|---|---|
| Model A | 0.1800 | 0.1400 | +0.0400 | Unfairly penalised |
| Model B | 0.2200 | 0.2200 | +0.0000 | Fairly scored |
| Model C | 0.1500 | 0.1100 | +0.0400 | Unfairly penalised |

A positive delta means the model was penalised for a valid alternative (e.g. said "14" but reference said "चौदह"). Lattice-WER corrects this. Models with zero delta were not penalised to begin with — their score stays unchanged.

---

## Recommended Timeline

| Day | Primary Focus | Key Output |
|---|---|---|
| 1 | Phase 0 setup + Q1 preprocessing | URL fixer working, dataset normalised and split |
| 2 | Q1 baseline WER on FLEURS, start LoRA training | Baseline WER number recorded |
| 3 | LoRA training running (GPU) + Q2 pipeline build | Number normaliser + English tagger complete |
| 4 | Q1 error analysis (steps d/e/f/g) | Error taxonomy, 25 sampled utterances, 1 fix implemented |
| 5 | Q3 frequency dict build + SymSpell classifier | Full 177k word list classified |
| 6 | Q3 low-confidence review + Q4 lattice design | Lattice documented, algorithm clear |
| 7 | Q4 full implementation + final packaging | Lattice-WER results, all deliverables polished |

---

## Deliverables Checklist

### Question 1
- [ ] Preprocessing steps documented (URL fix, torchaudio resampling, text normalisation)
- [ ] LoRA adapter saved (`./outputs/whisper-hi-lora/`)
- [ ] WER table: baseline vs. LoRA fine-tuned on FLEURS Hindi test set
- [ ] 25 systematically sampled error utterances (strategy described, not cherry-picked)
- [ ] Error taxonomy (3–5 examples per category: reference → output → cause)
- [ ] Top 3 fixes proposed with specific actionable steps
- [ ] At least 1 fix implemented with before/after WER on the error subset

### Question 2
- [ ] Number normalisation pipeline with 4–5 before/after examples from actual data
- [ ] 2–3 edge case examples with reasoning (e.g. दो-चार idiom guard explained)
- [ ] English word tagger with `[EN]...[/EN]` tagged output examples

### Question 3
- [ ] Hindi frequency dictionary built (Josh Talks transcripts + CC-100 or Leipzig)
- [ ] Loanword Devanagari whitelist populated
- [ ] Full 177k word classification complete (SymSpell three-phase pipeline)
- [ ] Final count of correctly vs. incorrectly spelled unique words
- [ ] Google Sheet: word | classification | confidence | reason
- [ ] Low-confidence bucket reviewed (40–50 words, accuracy analysis written up)
- [ ] 1–2 unreliable word categories identified with explanation

### Question 4
- [ ] Lattice construction approach explained (theory + bin system + consensus rule)
- [ ] Alignment unit justified (word-level, with reasoning)
- [ ] Lattice-WER computed for all 5 models
- [ ] Results table: standard WER vs. Lattice-WER vs. delta for each model
- [ ] Explanation of when model consensus overrides the human reference

---

*Prepared for Josh Talks AI Researcher Intern — Speech & Audio assignment.*
*Stack: LoRA + 8-bit quantization | torchaudio on-the-fly streaming | FastText `.ftz` (<1 MB) | SymSpell (milliseconds) | editdistance + JiWER*
