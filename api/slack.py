"""
Slack bot endpoint for Vercel (serverless).

Handles:
  - /competitor <name> <question>   -> slash command
  - @mentions in channels           -> event subscription (later phase)
  - URL verification handshake      -> Slack one-time setup ping

Knowledge base = competitor MDs on GitHub (raw.githubusercontent.com), fetched
on-demand. The derived analysis anchors (TL;DR, wins, losses, feature notes,
objections, pricing) are extracted and fed to Gemini together with the user's
question for a natural-language answer.
"""
import os
import re
import json
import hmac
import time
import hashlib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from http.server import BaseHTTPRequestHandler

import requests
from google import genai

# ---- Config from environment (set in Vercel project settings) ------------------
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_BOT_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")  # PAT with repo+workflow scope
GITHUB_REPO       = os.environ.get("GITHUB_REPO", "sunnykumaratwati/competitor-intel")
GITHUB_BRANCH     = os.environ.get("GITHUB_BRANCH", "main")
ALLOWED_CHANNEL   = os.environ.get("ALLOWED_CHANNEL", "")  # optional; restricts replies to one channel

MODEL = "gemini-2.5-flash"

# ---- Knowledge base helpers ----------------------------------------------------
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/competitors"

ANALYSIS_ANCHORS = ["tldr", "wins", "losses", "feature_notes", "objections", "pricing"]


def fetch_md(slug: str) -> str:
    """Pull competitor MD from GitHub. Returns '' if not found."""
    url = f"{RAW_BASE}/{slug}.md"
    try:
        r = requests.get(url, timeout=8)
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""


def trigger_seed_workflow(slug: str, url: str = "", docs_url: str = "") -> tuple:
    """Dispatch the GitHub Actions seed-on-demand workflow.
    Returns (ok_bool, message)."""
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not configured in Vercel env."
    api = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/seed-on-demand.yml/dispatches"
    try:
        r = requests.post(
            api,
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "ref": GITHUB_BRANCH,
                "inputs": {"slug": slug, "url": url, "docs_url": docs_url},
            },
            timeout=6,
        )
        if r.status_code == 204:
            return True, "ok"
        return False, f"GitHub API {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


def list_competitors() -> list:
    """Cheap list via GitHub contents API. Cached implicitly per cold-start."""
    try:
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/contents/competitors?ref={GITHUB_BRANCH}",
            timeout=8,
        )
        if r.status_code != 200:
            return []
        return [f["name"].removesuffix(".md") for f in r.json()
                if f.get("type") == "file" and f["name"].endswith(".md")]
    except Exception:
        return []


def extract_analysis(md: str) -> str:
    """Pull only the derived-analysis anchors from a competitor MD.
    Skips the bulky source-layer anchors — they're for the refresher, not the bot."""
    out_parts = []
    for name in ANALYSIS_ANCHORS:
        m = re.search(
            rf"<!-- ANCHOR:{name} -->(.*?)<!-- /ANCHOR:{name} -->",
            md, re.DOTALL,
        )
        if m:
            out_parts.append(m.group(1).strip())
    return "\n\n".join(out_parts)


# ---- Slack signature verification ---------------------------------------------
def verify_slack(headers: dict, body: bytes) -> bool:
    """Verify request came from Slack using HMAC + signing secret.
    Slack docs: https://api.slack.com/authentication/verifying-requests-from-slack"""
    if not SLACK_SIGNING_SECRET:
        return False
    ts = headers.get("x-slack-request-timestamp", "")
    sig = headers.get("x-slack-signature", "")
    if not ts or not sig:
        return False
    try:
        if abs(time.time() - int(ts)) > 60 * 5:  # >5min old = replay attack
            return False
    except ValueError:
        return False
    basestring = f"v0:{ts}:{body.decode('utf-8', errors='replace')}".encode()
    my_sig = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), basestring, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(my_sig, sig)


# ---- Gemini answer ------------------------------------------------------------
ANSWER_PROMPT = """You are Wati's competitive intelligence bot for sales reps.
Answer the sales rep's question using ONLY the battlecard content below.
Be direct, factual, and brief (3-6 sentences max unless the question demands more).
If the battlecard doesn't cover it, say so honestly — don't invent.

COMPETITOR: {slug}

BATTLECARD CONTENT:
{battlecard}

SALES REP QUESTION:
{question}

ANSWER:"""


def _gemini_call(slug: str, battlecard: str, question: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = ANSWER_PROMPT.format(slug=slug, battlecard=battlecard, question=question)
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return resp.text.strip()


def gemini_answer(slug: str, battlecard: str, question: str, hard_timeout: float = 2.2) -> tuple:
    """Try Gemini with a hard wall-clock timeout. Returns (text, used_gemini_bool).
    If Gemini is overloaded or slow, returns ('', False) so caller can fall back."""
    if not GEMINI_API_KEY:
        return "", False
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_gemini_call, slug, battlecard, question)
        try:
            return fut.result(timeout=hard_timeout), True
        except FutTimeout:
            print(f"[fallback] Gemini exceeded {hard_timeout}s timeout")
            return "", False
        except Exception as e:
            print(f"[fallback] Gemini error: {e}")
            return "", False


# --- Keyword-based section picker for the fallback path -------------------------
SECTION_KEYWORDS = [
    ("pricing",       ["pric", "cost", "$", "plan", "tier", "fee", "cheap", "expensive"]),
    ("wins",          ["win", "advantage", "better", "beat", "stronger", "why wati"]),
    ("losses",        ["lose", "lost", "weak", "miss", "lack", "gap", "worse"]),
    ("objections",    ["objection", "pushback", "concern", "worry", "complain"]),
    ("feature_notes", ["feature", "support", "integrat", "api", "limit", "do they", "can they"]),
    ("tldr",          []),  # default
]

SECTION_LABEL = {
    "tldr": "TL;DR vs Wati",
    "wins": "Where We Win",
    "losses": "Where We Lose",
    "feature_notes": "Feature Notes (from their docs)",
    "objections": "Objection Handling",
    "pricing": "Pricing Battle",
}


def pick_section(md: str, question: str) -> tuple:
    """Pick the most relevant analysis section based on question keywords.
    Returns (label, body_text)."""
    q = question.lower()
    for section, keywords in SECTION_KEYWORDS:
        if any(k in q for k in keywords):
            m = re.search(
                rf"<!-- ANCHOR:{section} -->(.*?)<!-- /ANCHOR:{section} -->",
                md, re.DOTALL,
            )
            if m:
                body = m.group(1).strip()
                # Trim section header (## Foo) since we'll prepend our own label
                body = re.sub(r"^##\s+[^\n]+\n+", "", body)
                return SECTION_LABEL[section], body[:2800]
    return "TL;DR", "_(no matching section)_"


# ---- Slash command handler ----------------------------------------------------
def handle_slash(form: dict) -> dict:
    """Synchronous handler that MUST return within Slack's 3s budget.
    Strategy: try Gemini with a hard 2.2s timeout; if it doesn't come back in
    time (or 503s), fall back to extracting the relevant MD section directly."""
    channel_id = form.get("channel_id", "")
    user_text  = (form.get("text") or "").strip()

    if ALLOWED_CHANNEL and channel_id != ALLOWED_CHANNEL:
        return {"response_type": "ephemeral",
                "text": ":lock: This bot only works in the configured channel."}

    if not user_text:
        comps = list_competitors()
        return {"response_type": "ephemeral",
                "text": ("Usage: `/competitor <name> <question>`\n"
                         f"Known competitors: {', '.join(comps) if comps else '(none yet)'}")}

    parts = user_text.split(maxsplit=1)
    slug = parts[0].lower()
    question = parts[1] if len(parts) > 1 else "Give me a quick TL;DR vs Wati."

    md = fetch_md(slug)
    if not md:
        # Unknown competitor — kick off the on-demand seed workflow.
        # User can also provide a URL inline: `/competitor add omnichat https://omnichat.com`
        url_hint = ""
        if slug == "add" and len(parts) > 1:
            # Form: /competitor add <slug> <url>
            tail = parts[1].split(maxsplit=2)
            if len(tail) >= 2:
                slug = tail[0].lower()
                url_hint = tail[1]
        ok, info = trigger_seed_workflow(slug, url=url_hint)
        if ok:
            return {"response_type": "in_channel",
                    "text": (f":hammer_and_wrench: I don't have a battlecard for *{slug}* yet — "
                             f"I've started building one. This takes 2-3 minutes. "
                             f"Try `/competitor {slug} <your question>` again shortly.")}
        comps = list_competitors()
        return {"response_type": "ephemeral",
                "text": (f":mag: I don't have a battlecard for `{slug}` yet, and auto-seed failed: {info}\n"
                         f"Known: {', '.join(comps) if comps else '(none)'}\n"
                         f"You can also seed manually: `/competitor add {slug} https://their-site.com`")}

    battlecard = extract_analysis(md)
    answer, used_gemini = gemini_answer(slug, battlecard, question)

    if used_gemini and answer:
        body = f"*{slug}* — {question}\n\n{answer}"
    else:
        label, section = pick_section(md, question)
        body = (f"*{slug}* — {question}\n"
                f"_(Gemini was slow/overloaded; serving the *{label}* section from the battlecard.)_\n\n"
                f"{section}")

    return {"response_type": "in_channel", "text": body}


# ---- Vercel entrypoint --------------------------------------------------------
class handler(BaseHTTPRequestHandler):  # noqa: N801 (Vercel expects lowercase)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # health check
        self._send_text(200, "competitor-intel slack bot OK")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        headers = {k.lower(): v for k, v in self.headers.items()}

        # Slack URL verification handshake (one-time, JSON body)
        ctype = headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_text(400, "bad json")
                return
            if payload.get("type") == "url_verification":
                self._send_json(200, {"challenge": payload.get("challenge", "")})
                return
            # Event callbacks (app_mention etc.) - placeholder for later phase
            if not verify_slack(headers, body):
                self._send_text(401, "bad signature")
                return
            self._send_json(200, {"ok": True})  # ack quickly; real handling later
            return

        # Slash command (form-encoded)
        if not verify_slack(headers, body):
            self._send_text(401, "bad signature")
            return

        form_pairs = urllib.parse.parse_qsl(body.decode("utf-8"))
        form = dict(form_pairs)
        response = handle_slash(form)
        self._send_json(200, response)
