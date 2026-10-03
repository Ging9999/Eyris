"""FinBERT headline sentiment (ProsusAI/finbert): shadow mode and research.

FinBERT is a free, open-source BERT model fine-tuned on financial text published
before 2020 (Financial PhraseBank). It has not seen the 2021-25 news, so unlike a
modern LLM it can be backtested honestly. It is deterministic on CPU (eval mode,
no sampling), so scores are reproducible for the review.

Live use is SHADOW ONLY: each round scores the same public headlines the news
veto sees and logs them to private/<round>/finbert_log.json. Nothing here
influences a decision. The dependencies (torch, transformers) are optional: if
they are missing, or anything fails, shadow mode reports why and the round carries on.
Research: scripts/finbert_experiment.py, reports/finbert_experiment.md.
"""
import json
from pathlib import Path

import numpy as np

MODEL = "ProsusAI/finbert"
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"   # pinned for reproducibility
MAX_TOKENS = 64          # headlines are short; truncation keeps CPU time small
_pipe = None


def available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval()
        labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
        _pipe = (tok, model, labels, torch)
    return _pipe


def score(titles, batch=64):
    """(n, 3) probabilities [positive, negative, neutral] for each headline."""
    titles = [str(t) for t in titles]
    if not titles:
        return np.zeros((0, 3))
    tok, model, labels, torch = _load()
    order = [labels.index(k) for k in ("positive", "negative", "neutral")]
    out = []
    with torch.no_grad():
        for i in range(0, len(titles), batch):
            enc = tok(titles[i:i + batch], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            out.append(probs[:, order])
    return np.vstack(out)


def sentiment(titles, batch=64):
    """Net sentiment in [-1, 1] per headline: P(positive) - P(negative)."""
    p = score(titles, batch)
    return p[:, 0] - p[:, 1] if len(p) else np.zeros(0)


def shadow(until, since, log_dir=None, collect=None):
    """Score the round's public headlines; returns a per-symbol summary. Never raises."""
    summary = {"model": MODEL, "since": str(since), "until": str(until)}
    try:
        if not available():
            summary["status"] = "skipped: torch/transformers not installed"
            return summary
        if collect is None:
            from .news import collect
        items, errors = collect(until, since)
        summary["feed_errors"] = len(errors)
        s = sentiment([x["title"] for x in items])
        per = {}
        for x, v in zip(items, s):
            per.setdefault(x["symbol"], []).append(float(v))
        summary["by_symbol"] = {k: {"n": len(v), "mean": float(np.mean(v)), "min": float(np.min(v))}
                                for k, v in sorted(per.items())}
        summary["status"] = f"ok: {len(items)} headlines"
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            rows = [{"symbol": x["symbol"], "published": str(x["published"]), "source": x["source"],
                     "title": x["title"], "sentiment": float(v)} for x, v in zip(items, s)]
            (Path(log_dir) / "finbert_log.json").write_text(json.dumps({**summary, "headlines": rows}, indent=1))
    except Exception as e:
        summary["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    return summary
