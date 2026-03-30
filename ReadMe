# Finetuning and Analysis of Whisper 🎙️🇮🇳

An end-to-end Automated Speech Recognition (ASR) pipeline specifically designed for processing, fine-tuning, and evaluating Hindi audio data from Josh Talks. This pipeline is built to handle raw real-world audio, automatically slice segments based on timestamps, compress via LoRA 8-bit quantization for consumer GPUs, and comprehensively normalize & evaluate Hindi text outputs using custom SymSpell heuristics and Lattice-WER scoring.

## 🚀 Features & Pipeline Stages

### Q1: Whisper LoRA Fine-Tuning (`q1_finetune.py`)
- **Smart Data Slicing:** Automatically downloads multi-hour Google Cloud audio streams into RAM and perfectly crops them into `<30s` Whisper training segments using JSON metadata timestamps.
- **Memory Efficiency:** Eliminates Out-Of-Memory (OOM) array crashes by streaming datasets directly to disk using `Dataset.from_generator()`.
- **4-Bit / 8-Bit Quantization:** Integrates `BitsAndBytesConfig` allowing massive Whisper-Small models to cleanly fit inside free Google Colab T4 GPUs (16GB VRAM limits).
- **LoRA Adapters:** Drastically cuts training time and preserves base model integrity by mapping Low-Rank Adaption modules exclusively to the Self-Attention (`q_proj`, `v_proj`) layers.

### Q2: Cleanup Pipeline (`q2_cleanup_pipeline.py`)
- Automatically detects and replaces explicit English characters dynamically via `fasttext`.
- Integrates extreme custom **Number Normalization**, intercepting and converting Hindi spoken compound digits (e.g. *चौदह*, *तीन सौ पचपन*) into pure numeric form (14, 355).
- Prevents Hindi loanwords natively adopted (but historically typed in English) from throwing hallucination triggers.

### Q3: Native Spell Check (`q3_spellcheck.py`)
- Injects a `SymSpell` dictionary mechanism with native Devanagari corpus support.
- Segregates confidence scores into `High`, `Medium`, and `Low` tiers.
- Specifically targets OCR/ASR generation artifacts to drastically catch contextually illogical word generations before WER evaluation.

### Q4: Lattice Word Error Rate (`q4_lattice_wer.py`)
- Systematically decodes outputs from Base Whisper vs Fine-tuned Whisper.
- Dynamically loads ground-truth evaluation references to benchmark models using standard JiWER scoring.

## 🛠 Required Usage Instructions
*The Excel metadata files and raw `.wav` Google Cloud dumps are strictly excluded from version control via `.gitignore` for data safety.*
You must place `FT Data.xlsx` into the `data/` directory natively before executing.

**Local Environment (CPU/Evaluation):**
```bash
pip install -r requirements.txt
python q2_cleanup_pipeline.py
python q3_spellcheck.py
python q4_lattice_wer.py
```

**Training Environment (Google Colab / Kaggle T4 GPU):**
Upload the entire codebase into a Python 3 Notebook environment equipped with Nvidia Hardware (T4/A100/L4, NOT standard TPUs).
```bash
!pip install -r requirements-colab.txt
!python q1_finetune.py
```

## 🏗 Dependencies & Tech Stack
- Transformers >= 4.41.0
- PEFT & BitsAndBytes (LoRA Compression)
- TorchAudio + Torch (Backend computation)
- FastText (Language Identification)
- SymSpellPy (Typo correction heuristics)
- Pandas & NumPy
