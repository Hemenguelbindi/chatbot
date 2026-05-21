#!/usr/bin/env python3
"""
Extract clean Russian text from Wikipedia XML dump.
Streaming: reads bz2 → parses XML → strips wiki markup → filters → saves.
"""

import bz2
import contextlib
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from lxml import etree

NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# ── Config ────────────────────────────────────────────────────────────────
DUMP_PATH = Path("/home/bindi/ruwiki.xml.bz2")
OUT_DIR = Path("data/wiki_extracted")
MIN_CHARS = 300  # min article length after cleaning
TRAIN_RATIO = 0.95
MAX_ARTICLES = 0  # 0 = no limit (for testing, set e.g. 1000)

# Files
TXT_TRAIN = OUT_DIR / "train.txt"
TXT_VALID = OUT_DIR / "valid.txt"
JSONL_ALL = OUT_DIR / "dataset.jsonl"

# ── Logging ───────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    filename=OUT_DIR / "extract.log",
    filemode="w",
)
log = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(console)

# ── Wiki cleanup rules ───────────────────────────────────────────────────

# Remove lines that are mostly wiki table syntax
TABLE_PATTERN = re.compile(r"^[!|{\-].*[!|}\-]\s*$")


def strip_wiki_markup(wiki_text: str) -> str:
    """Strip wiki markup using fast regex patterns instead of mwparserfromhell.
    Much faster for bulk processing. Removes templates, HTML tags, wiki links,
    references, tables, and other markup."""
    if not wiki_text or not wiki_text.strip():
        return ""

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", wiki_text, flags=re.DOTALL)

    # Remove <ref>...</ref>, <gallery>...</gallery>, <nowiki>...</nowiki>,
    # <pre>...</pre>, <source>...</source>, <syntaxhighlight>...</syntaxhighlight>
    # and other block tags
    text = re.sub(r"<(ref|gallery|nowiki|pre|source|syntaxhighlight|math|chem|timeline|imagemap|score|section|includeonly|noinclude|onlyinclude)[^>]*>.*?</\1\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove self-closing tags like <br/>, <hr/>, <br>, <hr>
    text = re.sub(r"<\s*(br|hr)\s*/?\s*>", "\n", text, flags=re.IGNORECASE)

    # Remove all remaining HTML-like tags (including unknown ones)
    text = re.sub(r"<[^>]*>", "", text)

    # Remove templates {{...}} — greedy with nesting support
    # Simple approach: remove templates that don't contain {{ or }}
    # For nested templates, use iterative removal
    old_len = -1
    while len(text) != old_len:
        old_len = len(text)
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Remove table markup {|...|}
    text = re.sub(r"\n\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove table row/cell markers
    text = re.sub(r"\n[!|][!|].*", "", text)

    # Convert wiki links: [[Article|display]] → display, [[Article]] → Article
    # But skip [[Category:...]], [[File:...]], [[Image:...]] (entirely remove)
    text = re.sub(r"\[\[(?:Категория|Category|Файл|File|Изображение|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # [[link|display]] → display
    text = re.sub(r"\[\[[^\[\]]*?\|([^\[\]]*?)\]\]", r"\1", text)
    # [[link]] → link
    text = re.sub(r"\[\[([^\[\]]*?)\]\]", r"\1", text)

    # External links: [http://... text] → text, [http://...] → remove
    text = re.sub(r"\[https?://[^\s\[\]]+\s+([^\[\]]*?)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\[\]]+\]", "", text)

    # Remove headings: = Heading = → Heading (keep text, remove markers)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = text.replace("'''", "").replace("''", "")

    # Remove magic words
    text = re.sub(r"__(NOTOC|NOEDITSECTION|FORCETOC|TOC|NEWSECTIONLINK|NONEWSECTIONLINK|NOGALLERY|HIDDENCAT|NOINDEX|INDEX)__", "", text)

    # Remove DEFAULTSORT and similar
    text = re.sub(r"\{\{DEFAULTSORT:[^}]*\}\}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{DEFAULTSORT\|[^}]*\}\}", "", text, flags=re.IGNORECASE)

    return text.strip()


def is_redirect(page_elem: etree.Element) -> bool:
    """Check if page is a redirect."""
    return page_elem.find(NS + "redirect") is not None


def get_namespace(page_elem: etree.Element) -> str:
    """Get namespace of the page."""
    ns_elem = page_elem.find(NS + "ns")
    return ns_elem.text if ns_elem is not None else ""


def get_text(page_elem: etree.Element) -> str:
    """Get revision text from page element."""
    rev = page_elem.find(NS + "revision")
    if rev is None:
        return ""
    text_elem = rev.find(NS + "text")
    return text_elem.text if text_elem is not None else ""


def get_title(page_elem: etree.Element) -> str:
    """Get page title."""
    title_elem = page_elem.find(NS + "title")
    return title_elem.text if title_elem is not None else ""


def clean_article(text: str) -> str:
    """Additional cleaning after wiki markup stripping."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip table lines
        if TABLE_PATTERN.match(line):
            continue
        # Skip empty/whitespace-only lines
        if not line:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Remove multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Remove lines that are just punctuation
    text = re.sub(r"\n[^\w]{1,5}\n", "\n", text)

    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_good_article(text: str) -> bool:
    """Check if the cleaned text is worth keeping."""
    if len(text) < MIN_CHARS:
        return False

    # Check if text is mostly non-Cyrillic (probably not Russian)
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF' or c == '\u0451' or c == '\u0401')
    if cyrillic_count < len(text) * 0.15:  # at least 15% Cyrillic chars
        return False

    return True


def process_dump():
    """Main pipeline: stream bz2 → parse XML → extract → filter → save."""
    articles = []
    stats = {
        "total_pages": 0,
        "redirects": 0,
        "non_main_ns": 0,
        "too_short": 0,
        "low_cyrillic": 0,
        "extraction_errors": 0,
        "kept": 0,
        "total_chars": 0,
    }

    log.info(f"Opening {DUMP_PATH} ...")
    log.info(f"Min article length: {MIN_CHARS} chars")
    start_time = time.time()

    # Use system bzcat/lbzip2 for faster decompression
    bzcat_cmd = None
    for cmd in ["lbzip2", "pbzip2", "bzip2"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
            bzcat_cmd = cmd + " -dc"
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if bzcat_cmd:
        log.info(f"Using {bzcat_cmd} for decompression")
        proc = subprocess.Popen(
            bzcat_cmd.split() + [str(DUMP_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        source = proc.stdout
    else:
        log.info("No system bzcat, falling back to Python bz2 module")
        source = bz2.open(DUMP_PATH, "rb")

    with contextlib.closing(source):
        context = etree.iterparse(
            source,
            events=("end",),
            tag=NS + "page",
        )

        for event, page_elem in context:
            stats["total_pages"] += 1

            if MAX_ARTICLES and stats["kept"] >= MAX_ARTICLES:
                break

            # Periodic progress
            if stats["total_pages"] % 10000 == 0:
                elapsed = time.time() - start_time
                rate = stats["total_pages"] / elapsed
                log.info(
                    f"  Pages: {stats['total_pages']:,}  "
                    f"Kept: {stats['kept']:,}  "
                    f"Rate: {rate:.0f} pg/s  "
                    f"Elapsed: {elapsed:.0f}s"
                )

            # Check redirect
            if is_redirect(page_elem):
                stats["redirects"] += 1
                page_elem.clear()
                continue

            # Check namespace
            ns = get_namespace(page_elem)
            if ns != "0":
                stats["non_main_ns"] += 1
                page_elem.clear()
                continue

            # Get wiki text
            wiki_text = get_text(page_elem)
            if not wiki_text or not wiki_text.strip():
                stats["extraction_errors"] += 1
                page_elem.clear()
                continue

            title = get_title(page_elem)

            # Skip very large articles (>500KB raw wiki text)
            if len(wiki_text) > 500_000:
                stats["too_short"] += 1  # reuse counter for "too large"
                page_elem.clear()
                continue

            # Log every 500th found article
            if stats["kept"] > 0 and stats["kept"] % 500 == 0:
                log.info(f"  Kept {stats['kept']:,} — last: {title}")

            # Strip wiki markup
            try:
                clean_text = strip_wiki_markup(wiki_text)
                clean_text = clean_article(clean_text)
            except Exception as e:
                log.info(f"  Skipped {repr(title)} — error: {e}")
                stats["extraction_errors"] += 1
                page_elem.clear()
                continue

            # Quality filters
            if not is_good_article(clean_text):
                if len(clean_text) < MIN_CHARS:
                    stats["too_short"] += 1
                else:
                    stats["low_cyrillic"] += 1
                page_elem.clear()
                continue

            stats["kept"] += 1
            stats["total_chars"] += len(clean_text)

            articles.append({
                "title": title,
                "text": clean_text,
                "chars": len(clean_text),
            })

            # Clear the element to free memory
            page_elem.clear()
            # Also clean up any ancestor elements
            while page_elem.getprevious() is not None:
                del page_elem.getparent()[0]

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("DONE — Processing complete!")
    log.info(f"Total XML pages scanned:  {stats['total_pages']:,}")
    log.info(f"  ↳ Redirects skipped:    {stats['redirects']:,}")
    log.info(f"  ↳ Non-main NS skipped:  {stats['non_main_ns']:,}")
    log.info(f"  ↳ Extraction errors:    {stats['extraction_errors']:,}")
    log.info(f"  ↳ Too short:            {stats['too_short']:,}")
    log.info(f"  ↳ Low Cyrillic:         {stats['low_cyrillic']:,}")
    log.info(f"  ↳ KEPT:                 {stats['kept']:,}")
    log.info(f"Total characters in kept:  {stats['total_chars']:,}")
    if stats["kept"] > 0:
        log.info(f"Avg article length:        {stats['total_chars'] // stats['kept']:,} chars")
    log.info(f"Time elapsed:              {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"Processing rate:           {stats['total_pages']/elapsed:.0f} pg/s")

    # ── Split train/valid ────────────────────────────────────────────────
    import random
    random.seed(42)
    random.shuffle(articles)

    split_idx = int(len(articles) * TRAIN_RATIO)
    train = articles[:split_idx]
    valid = articles[split_idx:]

    log.info("=" * 60)
    log.info(f"Train set: {len(train):,} articles")
    log.info(f"Valid set: {len(valid):,} articles")

    # ── Save TXT format (one article per line, for LM training) ──────────
    log.info("Saving train.txt ...")
    with open(TXT_TRAIN, "w", encoding="utf-8") as f:
        for article in train:
            f.write(article["text"].replace("\n", " ") + "\n")

    log.info("Saving valid.txt ...")
    with open(TXT_VALID, "w", encoding="utf-8") as f:
        for article in valid:
            f.write(article["text"].replace("\n", " ") + "\n")

    # ── Save JSONL format ────────────────────────────────────────────────
    log.info("Saving dataset.jsonl ...")
    with open(JSONL_ALL, "w", encoding="utf-8") as f:
        for article in articles:
            json.dump({"text": article["text"]}, f, ensure_ascii=False)
            f.write("\n")

    # ── Stats ────────────────────────────────────────────────────────────
    train_chars = sum(a["chars"] for a in train)
    valid_chars = sum(a["chars"] for a in valid)

    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info(f"train.txt:     {len(train):,} articles, {train_chars:,} chars,  ~{train_chars//len(train):,} avg")
    log.info(f"valid.txt:     {len(valid):,} articles, {valid_chars:,} chars,  ~{valid_chars//len(valid):,} avg")
    log.info(f"dataset.jsonl: {len(articles):,} articles (full set)")
    log.info(f"Files saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    process_dump()
