import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

from goemotions_benchmark.config import DATA_CACHE_DIR
from goemotions_benchmark.labels import EMOTION_NAMES

ALLOWED_EMOTIONS = set(EMOTION_NAMES)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


def _build_prompt(text: str) -> str:
    label_list = ", ".join(EMOTION_NAMES)
    return (
        "Classify the emotions expressed in this Reddit comment.\n"
        f"Valid labels (use exact spelling): {label_list}\n"
        'Return JSON only: {"emotions": ["label1", "label2"]}\n'
        "Include all labels that apply. Use [\"neutral\"] if no emotion is expressed.\n"
        f"Comment: {text}"
    )


def _extract_json(raw: str) -> dict | list | None:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\"emotions\"[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_emotion_response(raw: str) -> list[str]:
    """Parse Ollama output into validated GoEmotions labels."""
    payload = _extract_json(raw)
    candidates: list[str] = []

    if isinstance(payload, dict):
        value = payload.get("emotions", payload.get("labels", []))
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [str(item) for item in value]
    elif isinstance(payload, list):
        candidates = [str(item) for item in payload]

    if not candidates:
        lowered = raw.lower()
        for emotion in EMOTION_NAMES:
            if re.search(rf"\b{re.escape(emotion)}\b", lowered):
                candidates.append(emotion)

    validated = []
    seen = set()
    for label in candidates:
        clean = label.strip().lower()
        if clean in ALLOWED_EMOTIONS and clean not in seen:
            validated.append(clean)
            seen.add(clean)
    return validated


def _cache_file(cache_dir: Path, text: str) -> Path:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _call_ollama(prompt: str, model: str, base_url: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("response", "")


def check_ollama_available(base_url: str = DEFAULT_OLLAMA_URL) -> None:
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Ollama is not reachable. Install from https://ollama.com, start it, "
            f"and pull a model (e.g. `ollama pull {DEFAULT_OLLAMA_MODEL}`)."
        ) from exc


def predict_ollama(
    texts: list[str],
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> list[list[str]]:
    check_ollama_available(base_url)

    cache_dir = DATA_CACHE_DIR / "llm_cache" / model.replace(":", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[list[str]] = []
    for text in tqdm(texts, desc=f"Ollama ({model})"):
        cache_path = _cache_file(cache_dir, text)
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as handle:
                predictions.append(json.load(handle))
            continue

        prompt = _build_prompt(text)
        raw = _call_ollama(prompt, model=model, base_url=base_url)
        labels = parse_emotion_response(raw)

        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(labels, handle)

        predictions.append(labels)

    return predictions
