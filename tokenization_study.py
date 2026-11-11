# tokenization_study.py
import argparse
import csv
import html
import random
import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import TweetTokenizer, word_tokenize
from transformers import AutoTokenizer

SEED = 42
SAMPLE_SIZE = 5_000
INSPECTION_PER_DOMAIN = 15      # 15 tweets + 15 PubMed sentences = 30 examples
MAX_LEN_BIN = 20                # token lengths >= 20 chars go in the last bin

# --- custom tokenizer ---------------------------------------------------------
# NOTE: in re.VERBOSE mode "#" starts a comment, so the hashtag "#" is escaped.
CUSTOM_TWEET_RE = re.compile(
    r"""
    (?:https?://|www\.)\S*[^\s.,;:!?)\]"']     # URLs (no trailing punctuation)
    |(?<![\w@])@\w+                            # @mentions
    |(?<![\w\#])\#\w+                          # #hashtags
    |(?<!\w)(?:<3|[:;=8xX][\-^']?[)(/\\DPpOo|]|[)(/\\DPpOo|][\-^']?[:;=8xX])(?!\w)
                                               # emoticons (not inside words)
    |\w+(?:['’]\w+)*                           # words and contractions
    |([!?.,;:])\1*                             # runs of repeated punctuation
    |[^\w\s]                                   # any other symbol / emoji
    """,
    re.VERBOSE,
)


def custom_tweet_tokenize(text):
    # finditer + group(0): findall would return only the capture group.
    return [m.group(0) for m in CUSTOM_TWEET_RE.finditer(str(text))]
# --- end custom tokenizer -----------------------------------------------------


def whitespace_tokenize(text):
    return str(text).split()


def regex_tokenize(text):
    # General-purpose Unicode word/punctuation baseline.
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", str(text), flags=re.UNICODE)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_tweets(path):
    """Headerless Sentiment140 CSV (latin-1); tweet text is column index 5."""
    rows = []
    with open(path, "r", encoding="latin-1", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 6 and row[5].strip():
                rows.append(html.unescape(row[5]))   # &amp; &quot; &lt; -> real chars
    return rows


def load_pubmed_sentences(path):
    """PubMed RCT files: 'LABEL<TAB>sentence'; '###id' lines have no tab."""
    sentences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        _, sentence = line.split("\t", 1)
        if sentence.strip():
            sentences.append(sentence.strip())
    return sentences


def sample_texts(texts, n, seed):
    if len(texts) < n:
        raise ValueError(f"Need {n} records, found {len(texts)}.")
    return random.Random(seed).sample(texts, n)


# ----------------------------------------------------------------------------
# Tokenizers
# ----------------------------------------------------------------------------
def get_tokenizers():
    nltk.download("punkt_tab", quiet=True)   # needed by nltk.word_tokenize (new NLTK)
    nltk.download("punkt", quiet=True)       # older NLTK versions
    tweet_tok = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
    # Same rule-based tokenizer as en_core_web_sm, without needing the model download.
    nlp = spacy.blank("en")
    hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def nltk_tokenize(text):
        return word_tokenize(str(text))

    def spacy_tokenize(text):
        return [t.text for t in nlp.tokenizer(str(text))]

    def wordpiece_tokenize(text):
        return hf_tok.tokenize(str(text))

    return {
        "whitespace": whitespace_tokenize,
        "regex": regex_tokenize,
        "NLTK_word": nltk_tokenize,
        "TweetTokenizer": lambda t: tweet_tok.tokenize(str(t)),
        "spaCy": spacy_tokenize,
        "WordPiece": wordpiece_tokenize,
        "CustomTweetRegex": custom_tweet_tokenize,   # your designed tokenizer
    }


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def oov_rates(train_docs, eval_docs):
    """OOV w.r.t. a vocabulary built by the SAME tokenizer on the training half."""
    vocab = {t.casefold() for doc in train_docs for t in doc}
    flat = [t.casefold() for doc in eval_docs for t in doc]
    if not flat:
        return 0.0, 0.0
    types = set(flat)
    token_rate = sum(t not in vocab for t in flat) / len(flat)
    type_rate = sum(t not in vocab for t in types) / len(types)
    return token_rate, type_rate


TWEET_PATTERNS = {
    "URL": re.compile(r"https?://\S+|www\.\S+"),
    "mention": re.compile(r"(?<![\w@])@\w+"),
    "hashtag": re.compile(r"(?<![\w#])#\w+"),
    "emoticon": re.compile(
        r"(?<!\w)(?:<3|[:;=8xX][\-^']?[)(/\\DPpOo|]|[)(/\\DPpOo|][\-^']?[:;=8xX])(?!\w)"
    ),
    "contraction": re.compile(r"\w+['’]\w+"),
    "elongation/punct-run": re.compile(r"(\w)\1{2,}|[!?.]{2,}"),
}
PUBMED_PATTERNS = {
    "chemical": re.compile(
        r"\b\w*\d\w*-\w+|\b\w+-\d+|\b\w+\(\w+\)\w*|\b[A-Z][A-Za-z]*\d+[A-Za-z0-9]*\b"
    ),
    "abbreviation": re.compile(
        r"\b[A-Z]{2,}s?\b|\b(?:[A-Za-z]\.){2,}|\b(?:e\.g|i\.e|vs|et al)\."
    ),
    "numeric/stat": re.compile(
        r"\d+(?:\.\d+)?\s?%|\b[pP]\s?[<=>]\s?0?\.\d+|\d\s?(?:mg|kg|ml|mmol)\b"
    ),
    "hyphenated": re.compile(r"\b[A-Za-z]+-[A-Za-z]+\b"),
}
DOMAIN_PATTERNS = {"Sentiment140": TWEET_PATTERNS, "PubMed RCT": PUBMED_PATTERNS}


def pick_problem_examples(texts, tokenized, patterns, k):
    """Pick k docs likely to be problematic: contain a target phenomenon AND the
    tokenizers disagree on them. Round-robin over categories for coverage."""
    comparable = [m for m in tokenized if m != "WordPiece"]  # WordPiece always differs
    disagreement = [
        len({tuple(tokenized[m][i]) for m in comparable}) for i in range(len(texts))
    ]
    by_cat = {c: [] for c in patterns}
    for i, text in enumerate(texts):
        for c, pat in patterns.items():
            if pat.search(text):
                by_cat[c].append(i)
    for c in by_cat:
        by_cat[c].sort(key=lambda i: (-disagreement[i], i))

    chosen, seen = [], set()
    while len(chosen) < k and any(by_cat.values()):
        for c in patterns:
            while by_cat[c] and by_cat[c][0] in seen:
                by_cat[c].pop(0)
            if by_cat[c] and len(chosen) < k:
                i = by_cat[c].pop(0)
                seen.add(i)
                chosen.append((i, c))
    if len(chosen) < k:  # fill with the most contested remaining docs
        rest = sorted((i for i in range(len(texts)) if i not in seen),
                      key=lambda i: (-disagreement[i], i))
        chosen += [(i, "other") for i in rest[: k - len(chosen)]]
    return chosen


def summarize_domain(domain, texts, tokenizers):
    """Returns (summary_rows, inspection_rows, token_length_rows)."""
    half = len(texts) // 2          # first half = vocab for OOV, second half = test
    rows, length_rows, tokenized = [], [], {}

    for name, tokenize in tokenizers.items():
        docs = [tokenize(t) for t in texts]
        tokenized[name] = docs

        doc_lengths = np.array([len(d) for d in docs])
        flat = [tok for d in docs for tok in d]
        char_len = np.array([len(t) for t in flat]) if flat else np.array([0])
        oov_tok, oov_type = oov_rates(docs[:half], docs[half:])

        rows.append({
            "domain": domain,
            "tokenizer": name,
            "documents": len(docs),
            "mean_tokens_per_document": float(doc_lengths.mean()),
            "median_tokens_per_document": float(np.median(doc_lengths)),
            "mean_token_length_chars": float(char_len.mean()),
            "token_length_p50_chars": float(np.percentile(char_len, 50)),
            "token_length_p90_chars": float(np.percentile(char_len, 90)),
            "oov_rate_tokens": oov_tok,
            "oov_rate_types": oov_type,
            "unk_rate": flat.count("[UNK]") / len(flat) if flat else 0.0,
            "total_tokens": len(flat),
        })

        counts = Counter(np.minimum(char_len, MAX_LEN_BIN).tolist())
        for length in range(1, MAX_LEN_BIN + 1):
            length_rows.append({
                "domain": domain,
                "tokenizer": name,
                "token_length": length,
                "proportion": counts.get(length, 0) / max(len(flat), 1),
            })

    picks = pick_problem_examples(
        texts, tokenized, DOMAIN_PATTERNS.get(domain, TWEET_PATTERNS),
        min(INSPECTION_PER_DOMAIN, len(texts)),
    )
    inspection = []
    for i, category in picks:
        row = {"domain": domain, "issue_category": category, "text": texts[i]}
        for name, docs in tokenized.items():
            row[name] = " | ".join(docs[i])
        row["manual_error_notes"] = ""      # fill in by hand for the report
        inspection.append(row)

    return rows, inspection, length_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment140", required=True, help="Sentiment140 CSV")
    parser.add_argument("--pubmed-rct", required=True, help="PubMed RCT train.txt")
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--out", default="tokenization_results")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tweets = sample_texts(load_tweets(args.sentiment140), args.n, SEED)
    pubmed = sample_texts(load_pubmed_sentences(args.pubmed_rct), args.n, SEED + 1)

    tokenizers = get_tokenizers()
    summaries, inspections, lengths = [], [], []
    for domain, texts in [("Sentiment140", tweets), ("PubMed RCT", pubmed)]:
        r, e, l = summarize_domain(domain, texts, tokenizers)
        summaries += r
        inspections += e
        lengths += l

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame(inspections).to_csv(out / "manual_inspection_30.csv", index=False)
    pd.DataFrame(lengths).to_csv(out / "token_length_distribution.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
