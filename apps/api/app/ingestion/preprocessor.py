"""
TRUSTRAG — Text Preprocessing, Lexical Analysis, Zoning, and Stemming Pipeline.

Provides:
  - Text normalization & de-hyphenation (repairing PDF artifact breaks)
  - Canonical rule-based Porter Stemmer (Martin Porter, 1980)
  - Configurable stopwords (core grammatical + conversational query-noise filters)
  - Document Zoning (Title, Header, Summary, Body, Metadata) with zone-weight scoring
  - Multi-word N-Gram (Bigram) phrase extraction
"""

from __future__ import annotations

import functools
import re
import threading
import unicodedata
from enum import StrEnum

# ─── Contraction Mapping ──────────────────────────────────────────────────────

CONTRACTIONS: dict[str, str] = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

# Precompiled normalization regexes.
NONPRINTABLE_RE = re.compile(
    r"[\uf000-\uffff\u2022\u2023\u25cf\u25cb\u25aa\u25ab]"
)

HYPHEN_BREAK_RE = re.compile(
    r"(\w+)-\s*\n\s*(\w+)"
)

WHITESPACE_RE = re.compile(r"\s+")

# Match longest contractions first so "can't" is considered before "n't".
CONTRACTION_RE = re.compile(
    "|".join(
        re.escape(key)
        for key in sorted(CONTRACTIONS, key=len, reverse=True)
    )
)
TOKEN_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b")

HEADER_RE = re.compile(r"^(?:#+\s+|[A-Z0-9\s:\--]{4,50}\n)")

# ─── Standard IR & Query Noise Stopwords ─────────────────────────────────────

CORE_STOPWORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

# Conversational query fillers filtered out during query-time sparse search
# so search weights focus on topical content words rather than conversational wrappers
QUERY_NOISE_STOPWORDS: set[str] = {
    "tell",
    "please",
    "explain",
    "describe",
    "show",
    "summary",
    "summarize",
    "give",
    "detail",
    "details",
    "information",
    "document",
    "know",
    "like",
    "help",
    "provide",
    "list",
    "find",
    "mean",
    "meaning",
    "regard",
    "regarding",
    "discuss",
    "brief",
    "briefly",
}

STOPWORDS = CORE_STOPWORDS  # Default backward-compatible reference


def get_stopwords(include_query_noise: bool = False) -> set[str]:
    """Return stopword set, optionally including conversational query noise."""
    if include_query_noise:
        return CORE_STOPWORDS | QUERY_NOISE_STOPWORDS
    return CORE_STOPWORDS


# ─── Document Zoning ─────────────────────────────────────────────────────────


class ZoneType(StrEnum):
    """Document functional zones for Information Retrieval zone indexing and scoring."""

    TITLE = "title"
    HEADER = "header"
    SUMMARY = "summary"
    BODY = "body"
    METADATA = "metadata"


# Zone weight multiplier boosts for sparse BM25 index scoring
# Matches in Title and Header zones represent higher topical salience
ZONE_WEIGHT_BOOSTS: dict[str, float] = {
    ZoneType.TITLE.value: 2.0,
    ZoneType.HEADER.value: 1.5,
    ZoneType.SUMMARY.value: 1.3,
    ZoneType.BODY.value: 1.0,
    ZoneType.METADATA.value: 0.8,
}


def detect_chunk_zone(text: str, page: int = 1) -> str:
    """
    Detect the functional zone of a text chunk:
      - title: Page 1 document title or unit/chapter outline block
      - header: Section headers, topic titles, markdown headings
      - metadata: Administrative dates, ISBN, author information
      - summary: Abstract, summary, overview sections
      - body: Standard paragraph and explanatory content
    """
    text_stripped = text.strip()
    if not text_stripped:
        return ZoneType.BODY.value

    first_lines = text_stripped.split("\n")[:3]
    first_text = " ".join(first_lines).strip()
    first_upper = first_text.upper()

    # 1. Title Zone: Page 1 with course, unit, or syllabus markers
    if page == 1 and any(k in first_upper for k in ("UNIT-", "CHAPTER", "COURSE", "SYLLABUS")):
        return ZoneType.TITLE.value

    # 2. Metadata Zone: Effective dates, citations, policy headers
    meta_prefixes = ("effective from", "effective until", "author:", "isbn:", "doi:")
    if any(first_text.lower().startswith(p) for p in meta_prefixes):
        return ZoneType.METADATA.value

    # 3. Summary Zone
    summary_prefixes = ("summary:", "abstract:", "overview:", "executive summary:")
    if any(first_text.lower().startswith(p) for p in summary_prefixes):
        return ZoneType.SUMMARY.value

    # 4. Header Zone: Markdown # headers or all-caps topic lines (e.g. DATA STRUCTURES)
    if HEADER_RE.match(text_stripped):
        return ZoneType.HEADER.value

    return ZoneType.BODY.value


# ─── Text Normalization ───────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Perform lexical normalization on raw text:
      - NFKD Unicode normalization
      - Remove PDF/private-use bullet symbols
      - Repair hyphenated line breaks
      - Expand contractions in one regex pass
      - Normalize whitespace
    """
    if not text:
        return ""

    # Unicode normalization.
    normalized = unicodedata.normalize("NFKD", text)

    # Remove PDF artifacts / private-use symbols.
    normalized = NONPRINTABLE_RE.sub(" ", normalized)

    # Repair line-break hyphenations:
    # "infor-\nmation" -> "information"
    normalized = HYPHEN_BREAK_RE.sub(r"\1\2", normalized)

    # 4. Expand contractions
    text_lower = normalized.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = text_lower.replace(contraction, expansion)

    # 5. Collapse excessive horizontal whitespace, but PRESERVE line breaks.
    # Load-bearing: section/table heuristics (chunking strategies) and header
    # detection (detect_chunk_zone) split on "\n". Collapsing newlines to
    # spaces silently disables all of them — and token output is identical
    # either way since the lexer treats every whitespace run as a separator.
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", text_lower)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ─── Porter Stemmer Implementation ────────────────────────────────────────────


class PorterStemmer:
    """
    Canonical rule-based Porter Stemmer for English (Martin Porter, 1980).

    Reduces inflected and derived words to their base morphological stem.
    Guarantees consistent, stateless stemming for both index and query representations.
    """

    def __init__(self) -> None:
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i: int) -> bool:
        if self.b[i] in "aeiou":
            return False
        if self.b[i] == "y":
            if i == self.k0:
                return True
            return not self._cons(i - 1)
        return True

    def _m(self) -> int:
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        return any(not self._cons(i) for i in range(self.k0, self.j + 1))

    def _doublec(self, i: int) -> bool:
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        return ch not in "wxy"

    def _ends(self, s: str) -> bool:
        length = len(s)
        o = self.k - length + 1
        if o < self.k0:
            return False
        if self.b[o : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        length = len(s)
        o = self.j + 1
        self.b = self.b[:o] + s + self.b[o + length :]
        self.k = self.j + length

    def _r(self, s: str) -> None:
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self) -> None:
        if self.k <= self.k0:
            return
        c = self.b[self.k - 1]
        if c == "a":
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif c == "c":
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif c == "e":
            if self._ends("izer"):
                self._r("ize")
        elif c == "l":
            if self._ends("bli"):
                self._r("ble")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif c == "o":
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation") or self._ends("ator"):
                self._r("ate")
        elif c == "s":
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif c == "t":
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")
        elif c == "g" and self._ends("logi"):
            self._r("log")

    def _step3(self) -> None:
        if self.k <= self.k0:
            return
        c = self.b[self.k]
        if c == "e":
            if self._ends("icate"):
                self._r("ic")
            elif self._ends("ative"):
                self._r("")
            elif self._ends("alize"):
                self._r("al")
        elif c == "i":
            if self._ends("iciti"):
                self._r("ic")
        elif c == "l":
            if self._ends("ical"):
                self._r("ic")
            elif self._ends("ful"):
                self._r("")
        elif c == "s" and self._ends("ness"):
            self._r("")

    def _step4(self) -> None:
        if self.k <= self.k0:
            return
        c = self.b[self.k - 1]
        if c == "a":
            if not self._ends("al"):
                return
        elif c == "c":
            if not (self._ends("ance") or self._ends("ence")):
                return
        elif c == "e":
            if not self._ends("er"):
                return
        elif c == "i":
            if not self._ends("ic"):
                return
        elif c == "l":
            if not (self._ends("able") or self._ends("ible")):
                return
        elif c == "n":
            n_suffixes = ("ant", "ement", "ment", "ent")
            if not any(self._ends(s) for s in n_suffixes):
                return
        elif c == "o":
            if (self._ends("ion") and self.j >= self.k0 and self.b[self.j] in "st") or self._ends(
                "ou"
            ):
                pass
            else:
                return
        elif c == "s":
            if not self._ends("ism"):
                return
        elif c == "t":
            if not (self._ends("ate") or self._ends("iti")):
                return
        elif c == "u":
            if not self._ends("ous"):
                return
        elif c == "v":
            if not self._ends("ive"):
                return
        elif c == "z":
            if not self._ends("ize"):
                return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Stem a single word token."""
        word_clean = word.strip().lower()
        if len(word_clean) <= 2:
            return word_clean
        self.b = word_clean
        self.k = len(word_clean) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[self.k0 : self.k + 1]


# Use threading.local() so each thread has its own PorterStemmer instance.
# The stemmer has mutable instance state (self.b, self.k, etc.) that is not
# safe to share across threads — concurrent calls corrupt stemmer output.
_local = threading.local()


def _get_stemmer() -> PorterStemmer:
    """Return the current thread's PorterStemmer, creating it on first use."""
    if not hasattr(_local, "stemmer"):
        _local.stemmer = PorterStemmer()
    return _local.stemmer


# Bound LRU to 8k to prevent idle RSS creep (was 32768 — English vocab is smaller)
@functools.lru_cache(maxsize=8192)
def stem_word(word: str) -> str:
    """Convenience helper to stem a single word using thread-local PorterStemmer with LRU cache."""
    return _get_stemmer().stem(word)


# ─── N-Grams (Bigrams) ────────────────────────────────────────────────────────


def extract_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Generate n-gram compound phrases from a list of clean tokens."""
    if len(tokens) < n:
        return []
    return [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - n + 1)]


# ─── Lexical Analysis & Tokenization ──────────────────────────────────────────


def lexical_analyze(
    text: str,
    stem: bool = True,
    is_query: bool = False,
    include_bigrams: bool = False,
) -> list[str]:
    """
    Complete lexical analysis pipeline:
      1. Text Normalization (cleaning, de-hyphenation, contraction expansion)
      2. Token extraction (alphanumeric words, preserving hyphenated terms like 'n-gram')
      3. Stopword filtering (core or query-noise)
      4. Porter stemming (if stem=True)
      5. Optional bigram compound extraction (e.g. 'invert_file', 'data_structur')
    """
    clean_text = normalize_text(text)
    if not clean_text:
        return []

    # Match words and compound terms: letters, digits, and optional interior hyphens
    tokens = TOKEN_RE.findall(clean_text)

    # Use query noise filter when analyzing search queries
    stopwords = get_stopwords(include_query_noise=is_query)

    # Filter stopwords and very short noise (single character unless numeric/meaningful)
    filtered = [t for t in tokens if t not in stopwords and (len(t) > 1 or t.isdigit())]

    stemmed = [stem_word(t) for t in filtered] if stem else filtered

    if not include_bigrams or len(stemmed) < 2:
        return stemmed

    # Combine unigrams with bigrams for enhanced compound matching
    bigrams = extract_ngrams(stemmed, n=2)
    return stemmed + bigrams
