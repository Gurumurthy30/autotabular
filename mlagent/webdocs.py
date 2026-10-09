"""Official-docs fallback for the Docs agent. Only allowlisted documentation hosts are ever fetched
(error text and LLM output can never make us hit an arbitrary URL)."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import ClassVar

ALLOWED = {"scikit-learn.org", "lightgbm.readthedocs.io", "xgboost.readthedocs.io", "catboost.ai",
           "optuna.readthedocs.io", "pandas.pydata.org", "numpy.org", "docs.scipy.org"}
LOCAL_OK = {"127.0.0.1", "localhost"}        # plain http is only accepted for loopback (tests)

BASES = {"sklearn": "https://scikit-learn.org/stable", "lightgbm": "https://lightgbm.readthedocs.io/en/stable",
         "xgboost": "https://xgboost.readthedocs.io/en/stable", "catboost": "https://catboost.ai/docs/en",
         "optuna": "https://optuna.readthedocs.io/en/stable", "pandas": "https://pandas.pydata.org/docs",
         "numpy": "https://numpy.org/doc/stable", "scipy": "https://docs.scipy.org/doc/scipy"}


def _class_path(path: str) -> list[str]:
    """sklearn.ensemble.RandomForestClassifier.fit -> drop the trailing method (docs page = the class)."""
    parts = path.split(".")
    if len(parts) > 2 and parts[-1][:1].islower() and parts[-2][:1].isupper():
        parts = parts[:-1]
    return parts


def candidate_urls(lib: str, objects: list[str]) -> list[str]:
    base = BASES.get(lib)
    if not base:
        return []
    urls: list[str] = []
    for o in objects:
        parts = _class_path(o)
        dotted, name = ".".join(parts), parts[-1]
        urls.append({
            "sklearn": f"{base}/modules/generated/{dotted}.html",
            "lightgbm": f"{base}/pythonapi/{dotted}.html",
            # xgboost: link to the API page with a per-class anchor where possible
            "xgboost": f"{base}/python/python_api.html#{dotted}",
            # catboost: Python reference pages use class name in lowercase
            "catboost": f"{base}/concepts/python-reference_{name.lower()}.html",
            "optuna": f"{base}/reference/generated/{dotted}.html",
            "pandas": f"{base}/reference/api/{dotted}.html",
            "numpy": f"{base}/reference/generated/{dotted}.html",
            "scipy": f"{base}/reference/generated/{dotted}.html",
        }[lib])
    return list(dict.fromkeys(urls))


class _Text(HTMLParser):
    SKIP: ClassVar[frozenset[str]] = frozenset({"script", "style", "nav", "header", "footer", "aside", "noscript", "svg"})
    BLOCK: ClassVar[frozenset[str]] = frozenset({"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "pre", "dt", "dd", "section"})

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(html: str, max_chars: int = 6000) -> str:
    p = _Text()
    p.feed(html)
    text = re.sub(r"[ \t\r\f]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text[:max_chars]


def host_allowed(url: str) -> bool:
    u = urllib.parse.urlparse(url)
    host = u.hostname or ""
    if host in LOCAL_OK:
        return u.scheme in ("http", "https")
    return u.scheme == "https" and any(host == a or host.endswith("." + a) for a in ALLOWED)


def fetch_text(url: str, timeout: int = 15, max_chars: int = 6000) -> str | None:
    if not host_allowed(url):
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mlagent-docs/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read(400_000).decode("utf-8", "replace")
    except Exception:                                                  # noqa: BLE001 — offline / 404 / timeout
        return None
    text = html_to_text(raw, max_chars)
    return text if len(text) > 80 else None
