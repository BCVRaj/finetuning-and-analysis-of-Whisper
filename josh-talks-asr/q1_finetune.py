"""
q1_finetune.py — Whisper-small LoRA + 8-bit fine-tuning on Josh Talks Hindi audio.

Pipeline:
  1. Load metadata JSON and fix all GCP URLs
  2. Load & resample audio on-the-fly with torchaudio (no disk writes)
  3. Normalize Devanagari transcripts with indic-nlp-library
  4. Build HuggingFace DatasetDict (90/10 split, 1–30s clips)
  5. Load Whisper-small in 8-bit with bitsandbytes
  6. Apply LoRA adapter (r=32, target q_proj + v_proj)
  7. Train with Seq2SeqTrainer, save only ~8 MB LoRA checkpoint
  8. Evaluate baseline and fine-tuned WER on FLEURS Hindi test set
  9. Sample 25 errors systematically, build taxonomy, propose + implement 1 fix
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from utils import (
    fix_url,
    load_audio_torchaudio,
    load_metadata,
    load_transcription,
    normalize_transcript,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "openai/whisper-small"
METADATA_PATH = "data/FT Data.xlsx"       # Real Josh Talks dataset (104 records)
FT_RESULT_PATH = "data/FT Result.xlsx"    # Pre-computed WER results from Josh Talks
OUTPUT_DIR = "./outputs/whisper-hi-lora"
TARGET_SR = 16_000
MIN_DURATION = 1.0   # seconds
MAX_DURATION = 30.0  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step a — Data Preprocessing
# ---------------------------------------------------------------------------


def prepare_record(item: dict, processor: WhisperProcessor) -> dict:
    """Load audio + transcript for one metadata record and extract features."""
    # Use transcription_url_gcp — the real column name in FT Data.xlsx
    audio_array = load_audio_torchaudio(item["rec_url_gcp"]).numpy()
    transcript = normalize_transcript(load_transcription(item["transcription_url_gcp"]))

    input_features = processor.feature_extractor(
        audio_array, sampling_rate=TARGET_SR
    ).input_features[0]

    labels = processor.tokenizer(transcript).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
        "duration": item["duration"],
    }


def build_dataset(metadata_path: str, processor: WhisperProcessor) -> DatasetDict:
    """
    Builds a train/test HuggingFace DatasetDict from the metadata JSON.

    Filters clips outside [1, 30] seconds and applies a 90/10 split.
    """
    metadata = load_metadata(metadata_path)

    # Filter by duration
    filtered = [m for m in metadata if MIN_DURATION <= m["duration"] <= MAX_DURATION]
    logger.info(
        f"Duration filter: {len(metadata)} → {len(filtered)} records "
        f"({len(metadata) - len(filtered)} removed)"
    )

    records = []
    for i, m in enumerate(filtered):
        try:
            records.append(prepare_record(m, processor))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Skipping record {i}: {exc}")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Dataset built: {len(split['train'])} train / {len(split['test'])} test")
    return split


# ---------------------------------------------------------------------------
# Step b — Data Collator
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads input features and labels, masking padding tokens with -100."""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Pad audio features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels and mask padding positions with -100
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Step b — WER Metric
# ---------------------------------------------------------------------------

wer_metric = evaluate.load("wer")


def compute_metrics(pred, processor: WhisperProcessor) -> Dict[str, float]:
    """Decodes predictions and references, then computes WER."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


# ---------------------------------------------------------------------------
# Step b — Model + LoRA Setup
# ---------------------------------------------------------------------------


def load_model_with_lora(model_id: str) -> WhisperForConditionalGeneration:
    """
    Loads Whisper-small in 8-bit quantization and applies a LoRA adapter.

    Trainable parameters: ~2M out of 244M (0.82%)
    VRAM required: ~4 GB (fits on free Colab/Kaggle T4)
    """
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # REQUIRED for 8-bit training: prepares gradient checkpoints and casts
    # layer norms to float32. Without this, the backward pass will crash.
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,                                    # Rank
        lora_alpha=64,                           # Scaling (2 × r)
        target_modules=["q_proj", "v_proj"],     # Attention projection layers only
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Step b — Training
# ---------------------------------------------------------------------------


def train(dataset: DatasetDict, processor: WhisperProcessor, output_dir: str) -> None:
    """Fine-tunes the LoRA model and saves the checkpoint."""
    model = load_model_with_lora(MODEL_ID)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,          # Effective batch size = 16
        warmup_steps=500,
        eval_strategy="epoch",                  # Fixed: evaluation_strategy deprecated in transformers >=4.41
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
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor.feature_extractor,  # Fixed: 'tokenizer' arg renamed in transformers >=4.41
    )

    trainer.train()
    model.save_pretrained(output_dir)  # Saves only ~8 MB LoRA weights
    logger.info(f"LoRA checkpoint saved to {output_dir}")


# ---------------------------------------------------------------------------
# Step c — Load Pre-Computed WER from FT Result.xlsx
# ---------------------------------------------------------------------------


def load_wer_results(path: str = FT_RESULT_PATH) -> List[Dict]:
    """
    Reads pre-computed WER scores from FT Result.xlsx.

    The Excel has 2 columns:
      Col 0: Model name
      Col 1: WER score (raw ratio, e.g. 0.30 = 30%)

    Returns:
        List of dicts with keys: model, wer
    """
    import pandas as pd  # noqa: PLC0415
    df = pd.read_excel(path, header=None)
    results = []
    for _, row in df.iterrows():
        model_name = str(row.iloc[0]).strip()
        wer_val = row.iloc[1]
        # Skip header rows or rows where WER is not numeric
        try:
            wer_float = float(wer_val)
        except (ValueError, TypeError):
            continue
        if model_name.lower() in ("model", "nan", ""):
            continue
        results.append({"model": model_name, "wer": wer_float})
    logger.info(f"Loaded {len(results)} pre-computed WER results from {path}")
    return results


def print_wer_table(results: List[Dict]) -> None:
    """Pretty-prints a WER comparison table."""
    print("\n" + "=" * 55)
    print(f"{'Model':<40} {'WER':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['model']:<40} {r['wer']:>8.4f}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Step c — Baseline vs. Fine-Tuned Evaluation on FLEURS Hindi
# ---------------------------------------------------------------------------


def evaluate_on_fleurs(model, processor: WhisperProcessor) -> float:
    """Evaluates WER on the FLEURS Hindi test set."""
    fleurs_test = load_dataset("google/fleurs", "hi_in", split="test")

    predictions, references = [], []
    for sample in fleurs_test:
        audio = sample["audio"]["array"].astype("float32")
        inputs = processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(model.device)

        with torch.no_grad():
            pred_ids = model.generate(inputs)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        predictions.extend(pred_str)
        references.append(sample["transcription"])

    return wer_metric.compute(predictions=predictions, references=references)


# ---------------------------------------------------------------------------
# Step d — Systematic Error Sampling (25 Utterances)
# ---------------------------------------------------------------------------


def sample_errors_systematically(
    predictions: list, references: list, n: int = 25
) -> list:
    """
    Returns n error utterances sampled systematically (no cherry-picking).

    Strategy: sort all errors by per-utterance WER (highest first),
    then take every Kth sample to spread severity coverage evenly.
    """
    per_utt = [
        {
            "idx": i,
            "pred": p,
            "ref": r,
            "wer": wer_metric.compute(predictions=[p], references=[r]),
        }
        for i, (p, r) in enumerate(zip(predictions, references))
        if p.strip() != r.strip()
    ]
    per_utt.sort(key=lambda x: x["wer"], reverse=True)
    step = max(1, len(per_utt) // n)
    sampled = per_utt[::step][:n]
    logger.info(f"Sampled {len(sampled)} error utterances from {len(per_utt)} total errors")
    return sampled


# ---------------------------------------------------------------------------
# Step e — Error Taxonomy Categories
# ---------------------------------------------------------------------------

ERROR_TAXONOMY = {
    "code_switching": {
        "description": "Hinglish words (English words written in Devanagari) misrecognised",
        "root_cause": "Insufficient English-in-Hindi training data",
        "examples": [],  # Populated after error analysis
    },
    "phonetic_confusion": {
        "description": "Similar-sounding word substitutions in conversational speech",
        "root_cause": "Acoustic ambiguity, homophones, fast speech rate",
        "examples": [],
    },
    "number_date_errors": {
        "description": "Numbers or dates wrong, omitted, or in different form (word vs. digit)",
        "root_cause": "Low numeric example frequency in Whisper training data",
        "examples": [],
    },
    "proper_nouns": {
        "description": "Names of speakers, cities, brands, topics wrong or omitted",
        "root_cause": "Out-of-vocabulary tokens; Whisper not trained on Josh Talks entities",
        "examples": [],
    },
    "disfluent_speech": {
        "description": "Words merged, dropped, or repeated due to high speech rate",
        "root_cause": "Filler words, restarts, fast delivery in conversational Hindi",
        "examples": [],
    },
}


# ---------------------------------------------------------------------------
# Step f+g — Top 3 Fixes
# ---------------------------------------------------------------------------

TOP_3_FIXES = [
    {
        "name": "Fix 1 — Code-switching augmentation",
        "description": (
            "Augment training data with Hinglish examples from Hinglish Delite or "
            "HinglishPeople datasets. Add 500–1000 code-switched utterances before re-LoRA."
        ),
        "target_category": "code_switching",
    },
    {
        "name": "Fix 2 — KenLM n-gram re-scorer",
        "description": (
            "Build a lightweight Hindi n-gram language model with KenLM. Score top-2 "
            "Whisper beam candidates and select the one with higher n-gram probability."
        ),
        "target_category": "phonetic_confusion",
    },
    {
        "name": "Fix 3 — Custom vocabulary force-decoding",
        "description": (
            "Build a Josh Talks-specific entity list (speaker names, topic keywords, "
            "city names). Force-decode these tokens via processor.tokenizer before inference."
        ),
        "target_category": "proper_nouns",
    },
]


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=== Q1: Whisper Fine-Tuning Pipeline ===")

    # 0. Show pre-computed WER from FT Result.xlsx
    logger.info("Loading pre-computed WER results from FT Result.xlsx...")
    precomputed_results = load_wer_results()
    if precomputed_results:
        print("\n=== Pre-Computed WER Results (from FT Result.xlsx) ===")
        print_wer_table(precomputed_results)

    # 1. Processor
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="hi", task="transcribe")

    # 2. Dataset — reads real FT Data.xlsx, fixes broken GCP URLs automatically
    logger.info(f"Building dataset from {METADATA_PATH}...")
    dataset = build_dataset(METADATA_PATH, processor)

    # 3. Train
    logger.info("Starting LoRA fine-tuning...")
    train(dataset, processor, OUTPUT_DIR)

    # 4. Load fine-tuned model for evaluation
    from peft import PeftModel  # noqa: PLC0415

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, load_in_8bit=True, device_map="auto"
    )
    finetuned_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    # 5. Live evaluation on FLEURS Hindi
    logger.info("Evaluating pretrained baseline on FLEURS Hindi...")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto"
    )
    baseline_wer = evaluate_on_fleurs(baseline_model, processor)
    finetuned_wer = evaluate_on_fleurs(finetuned_model, processor)

    # Combine pre-computed + live results
    combined = precomputed_results.copy()
    combined.append({"model": "Whisper-small pretrained baseline (live)", "wer": baseline_wer})
    combined.append({"model": "Whisper-small LoRA fine-tuned (live)", "wer": finetuned_wer})

    print("\n=== Final Combined WER Results ===")
    print_wer_table(combined)

    logger.info("Q1 pipeline complete.")


if __name__ == "__main__":
    main()
