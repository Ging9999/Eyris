"""Reduce-only LLM news veto (TradingAgents / FinCon "risk manager" role).

Before each round: collect public headlines (Yahoo Finance RSS) and SEC 8-K
filings published before the submission deadline, ask an approved model
(Claude Opus 5) which names face a *new, material downside* risk, and turn its
answer into trims {asset: cut}. The veto can only reduce weights; any failure
(no key, network, refusal, bad output) means "no veto", never an error.

Every prompt and response is logged (and cached by prompt hash) for the
competition's LLM disclosure and reproducibility requirements.
"""
import hashlib
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import UNIVERSE

ET_TZ = ZoneInfo("America/New_York")
MODEL = "claude-opus-5"   # on the competition's approved list (Opus 5 / Sonnet 5 / Haiku 4.5)
EFFORT = "medium"
CUTS = {"none": 0.0, "elevated": 0.5, "severe": 1.0}
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "private" / "llm_cache"
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
SEC_ATOM = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sym}&type=8-K"
            "&dateb=&owner=include&count=10&output=atom")

SYSTEM = """You are the risk manager of a long-only, low-turnover equity portfolio of 30 large US stocks.
Before the next trade you receive the public headlines and SEC 8-K filings for these stocks published since the last decision.
Your only power is to REDUCE a position. Flag a stock only for a new, specific, material event that creates meaningful downside risk before the next few trading sessions, for example: a guidance cut or large earnings miss, a regulatory or legal action, a product recall or safety incident, a sudden executive departure, a credit or liquidity problem, or a major cyber incident.
Do not flag routine news, analyst price-target chatter, market commentary, positive news, or events that already happened and were priced in days ago. When in doubt, answer "none". Most stocks on most days should be "none".
Use only the information provided. Do not rely on any knowledge of later events."""

SCHEMA = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "enum": list(UNIVERSE)},
                    "risk": {"type": "string", "enum": ["none", "elevated", "severe"]},
                    "reason": {"type": "string"},
                },
                "required": ["symbol", "risk", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["flags"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- collection
def _get(url, headers=None, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def yahoo_headlines(sym):
    root = ET.fromstring(_get(YAHOO_RSS.format(sym=sym)))
    out = []
    for it in root.iter("item"):
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone(ET_TZ)
        except (TypeError, ValueError):
            continue
        out.append({"symbol": sym, "source": "Yahoo Finance", "published": ts,
                    "title": (it.findtext("title") or "").strip(),
                    "text": re.sub(r"\s+", " ", it.findtext("description") or "").strip()[:600]})
    return out


def sec_8k(sym):
    ua = os.environ["SEC_USER_AGENT"]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(_get(SEC_ATOM.format(sym=sym), {"User-Agent": ua}))
    out = []
    for e in root.findall("a:entry", ns):
        try:
            ts = datetime.fromisoformat(e.findtext("a:updated", namespaces=ns)).astimezone(ET_TZ)
        except (TypeError, ValueError):
            continue
        summary = re.sub(r"<[^>]+>", " ", e.findtext("a:summary", default="", namespaces=ns))
        out.append({"symbol": sym, "source": "SEC 8-K", "published": ts,
                    "title": (e.findtext("a:title", namespaces=ns) or "").strip(),
                    "text": re.sub(r"\s+", " ", summary).strip()[:600]})
    return out


def collect(until, since, symbols=UNIVERSE, fetchers=None):
    """Items with since <= published < until (strictly before the deadline).

    SEC requires a contact in the User-Agent, so 8-Ks are fetched only when
    SEC_USER_AGENT is set (e.g. "Your Name you@example.com").
    """
    if fetchers is None:
        fetchers = (yahoo_headlines, sec_8k) if os.environ.get("SEC_USER_AGENT") else (yahoo_headlines,)
    items, errors = [], []
    for sym in symbols:
        for f in fetchers:
            try:
                items += [x for x in f(sym) if since <= x["published"] < until]
            except Exception as e:  # one feed failing must not block the round
                errors.append(f"{f.__name__}({sym}): {type(e).__name__}")
    items.sort(key=lambda x: (x["symbol"], x["published"], x["title"]))
    return items, errors


# --------------------------------------------------------------------------- LLM call
def build_prompt(items, until):
    lines = [f"Decision deadline: {until:%Y-%m-%d %H:%M} ET. Items published before it:", ""]
    for x in items:
        lines.append(f"[{x['symbol']}] {x['published']:%Y-%m-%d %H:%M} {x['source']}: {x['title']}"
                     + (f" | {x['text']}" if x["text"] else ""))
    lines += ["", "Return one entry per stock you flag as elevated or severe (omit stocks with no material risk)."]
    return "\n".join(lines)


def ask_model(prompt, client=None):
    """Structured call to the approved model. Returns (flags, raw_text, meta)."""
    import anthropic
    client = client or anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    meta = {"model": resp.model, "stop_reason": resp.stop_reason, "request_id": resp._request_id,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}}
    if resp.stop_reason == "refusal":
        return [], "", meta
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["flags"], text, meta


def veto(until, since=None, log_dir=None, client=None, fetch=collect):
    """Reduce-only trims {asset_index: cut} for a round with deadline ``until``."""
    since = since or until - timedelta(hours=18)
    log = {"model": MODEL, "effort": EFFORT, "system": SYSTEM, "since": str(since), "until": str(until)}
    trims = {}
    try:
        items, errors = fetch(until, since)
        log.update(n_items=len(items), feed_errors=errors)
        if not items:
            log["result"] = "no items"
            return trims, log
        prompt = build_prompt(items, until)
        key = hashlib.sha256((MODEL + EFFORT + SYSTEM + prompt).encode()).hexdigest()
        log.update(prompt=prompt, prompt_sha256=key)
        cached = CACHE / f"{key}.json"
        if cached.exists():
            hit = json.loads(cached.read_text())
            flags, text, meta = hit["flags"], hit["text"], {**hit["meta"], "cache_hit": True}
        else:
            flags, text, meta = ask_model(prompt, client)
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"flags": flags, "text": text, "meta": meta}))
        log.update(response=text, meta=meta)
        for f in flags:
            cut = CUTS.get(f.get("risk"), 0.0)
            if cut > 0 and f.get("symbol") in UNIVERSE:
                i = UNIVERSE.index(f["symbol"])
                trims[i] = max(trims.get(i, 0.0), cut)
        log["trims"] = {UNIVERSE[i]: c for i, c in trims.items()}
    except Exception as e:  # never block a round on the LLM
        log["error"] = f"{type(e).__name__}: {e}"
        trims = {}
    finally:
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / "llm_log.json").write_text(json.dumps(log, indent=1, default=str))
    return trims, log
