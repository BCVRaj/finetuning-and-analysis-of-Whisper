"""
utils.py — Shared helpers for the Josh Talks ASR pipeline.

Contains:
- fix_url(): Rewrites legacy GCP URLs to the upload_goai format (CRITICAL — run this first)
- load_metadata(): Downloads and fixes all URLs in the dataset metadata JSON
- load_transcription(): Fetches the transcription text from a GCP JSON file
- load_hindi_wordlist(): Loads a Hindi word list file into a set for O(1) lookups
- load_audio_torchaudio(): Streams audio from a GCP URL, resamples to 16kHz on-the-fly
- normalize_transcript(): Normalises Devanagari script text to clean Unicode NFC form
"""

import io
import json
import re
import unicodedata
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import requests
import torch
import torchaudio
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL Fixer — implement this FIRST; all data access depends on it
# ---------------------------------------------------------------------------

_OLD_BUCKETS = (
    "storage.googleapis.com/goai_audio",
    "storage.googleapis.com/joshtalks-data-collection/hq_data/hi",
)
_NEW_BUCKET = "storage.googleapis.com/upload_goai"


def fix_url(old_url: str) -> str:
    """
    Rewrites legacy GCP URLs to the upload_goai format.

    Handles both the published legacy format and the actual format found in
    FT Data.xlsx.

    Example:
      Input:  https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_audio.wav
      Output: https://storage.googleapis.com/upload_goai/967179/825780_audio.wav
    """
    for old_bucket in _OLD_BUCKETS:
        if old_bucket in old_url:
            return old_url.replace(old_bucket, _NEW_BUCKET)
    return old_url


# ---------------------------------------------------------------------------
# Metadata Loading
# ---------------------------------------------------------------------------


# URL columns that exist in the real FT Data.xlsx
_URL_COLUMNS = ("rec_url_gcp", "transcription_url_gcp", "metadata_url_gcp")


def load_metadata(path: str) -> list:
    """
    Loads dataset metadata from FT Data.xlsx (or legacy JSON) and fixes all
    broken GCP URLs in-place.

    Real Excel column names: rec_url_gcp, transcription_url_gcp, metadata_url_gcp

    Args:
        path: Path to FT Data.xlsx (or a legacy .json file).

    Returns:
        List of record dicts with all URLs rewritten to the upload_goai format.
    """
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        records = df.to_dict(orient="records")
    else:
        # Legacy JSON path — kept for backward compatibility
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

    fixed_count = 0
    for record in records:
        for key in _URL_COLUMNS:
            val = record.get(key)
            if isinstance(val, str) and any(old_b in val for old_b in _OLD_BUCKETS):
                record[key] = fix_url(val)
                fixed_count += 1

    logger.info(f"Loaded {len(records)} records from {path}; fixed {fixed_count} URLs.")
    return records


# ---------------------------------------------------------------------------
# Transcription Fetching
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = 30  # seconds


def load_transcription_segments(transcription_url: str) -> list:
    """
    Fetches the transcription segments from a GCP JSON file.

    Returns:
        List of dicts: [{"start": 0.0, "end": 2.5, "text": "..."}, ...]
    """
    try:
        response = requests.get(fix_url(transcription_url), timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        # Fallback if it's not a list for some reason
        return []
    except requests.RequestException as exc:
        logger.warning(f"Failed to fetch transcription from {transcription_url}: {exc}")
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"Invalid JSON in transcription response: {exc}")
        return []


# ---------------------------------------------------------------------------
# Hindi Wordlist Loading
# ---------------------------------------------------------------------------


def load_hindi_wordlist(path: str) -> set:
    """
    Loads a Hindi word list file into a frozenset for O(1) lookups.

    File format: one word per line, UTF-8 encoded.

    Args:
        path: Path to the word list file.

    Returns:
        A set of Hindi words (strings).
    """
    try:
        with open(path, encoding="utf-8") as f:
            words = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(words)} words from {path}")
        return words
    except FileNotFoundError:
        logger.warning(f"Wordlist not found: {path}. Returning empty set.")
        return set()


# ---------------------------------------------------------------------------
# Audio Loading — torchaudio on-the-fly streaming
# ---------------------------------------------------------------------------

TARGET_SR = 16_000  # Whisper expects 16kHz mono


def load_audio_torchaudio(url: str) -> torch.Tensor:
    """
    Streams audio from a GCP URL, resamples to 16kHz mono on-the-fly.

    No files are saved to disk — everything is processed in memory.

    Args:
        url: GCP audio URL (must already be in upload_goai format).

    Returns:
        1D float32 tensor of shape (num_samples,) at 16kHz, ready for
        Whisper's feature extractor.

    Raises:
        requests.RequestException: If the network request fails.
        RuntimeError: If torchaudio cannot decode the audio.
    """
    response = requests.get(url, stream=True, timeout=_HTTP_TIMEOUT)
    response.raise_for_status()

    audio_bytes = io.BytesIO(response.content)
    waveform, orig_sr = torchaudio.load(audio_bytes)

    # Convert stereo / multi-channel to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if the source sample rate differs from 16kHz
    if orig_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform.squeeze()  # Shape: (num_samples,)


# ---------------------------------------------------------------------------
# Transcript Normalisation
# ---------------------------------------------------------------------------

_normalizer = IndicNormalizerFactory().get_normalizer("hi")
_DEVANAGARI_RE = re.compile(r"[^\u0900-\u097F\s]")


def normalize_transcript(text: str) -> str:
    """
    Normalises a raw Hindi transcript to clean Unicode form.

    Steps:
      1. Devanagari script normalisation via indic-nlp-library
      2. Unicode NFC composition
      3. Strip all non-Devanagari characters (keeps spaces)
      4. Collapse consecutive whitespace

    Args:
        text: Raw transcript string.

    Returns:
        Normalised transcript safe for Whisper tokenisation.
    """
    text = _normalizer.normalize(text)
    text = unicodedata.normalize("NFC", text)
    text = _DEVANAGARI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # URL fixer test
    old = "https://storage.googleapis.com/goai_audio/967179/825780.wav"
    fixed = fix_url(old)
    assert "upload_goai" in fixed, f"URL fix failed: {fixed}"
    assert "goai_audio" not in fixed, f"Old bucket still present: {fixed}"
    print(f"✓ fix_url: {old}\n       → {fixed}")

    # Transcript normalisation test
    sample = "उसने तीन सौ चौवन\u0902 किताबें खरीदीं।"
    normed = normalize_transcript(sample)
    print(f"✓ normalize_transcript: '{sample}' → '{normed}'")

    print("\nAll utils.py self-tests passed ✓")
