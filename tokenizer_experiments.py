"""
Tokenizer experiments for: Comparative Study of Tokenization Methods on
Social Media and Biomedical Text.

What this script does
  1. Loads Sentiment140 (robust encoding, HTML entities un-escaped, rows with
     corrupted characters removed) and PubMed RCT (optionally detokenized).
  2. Runs seven tokenizers (whitespace, regex, NLTK word_tokenize,
     TweetTokenizer, spaCy, WordPiece = bert-base-uncased WITHOUT [CLS]/[SEP],
     and CustomTweetRegex) on 5,000 docs per domain for several seeds.
  3. Saves: tokens/doc, OOV, avg token length (mean +- std), token length
     distribution, "units kept whole" (URL, mention, hashtag, emoticon,
     decimal, hyphenated, abbreviation, acronym, chemical, contraction,
     punctuation run, elongation, +/-), a CustomTweetRegex ablation,
     optional downstream sentiment classification, and run_config.json
     (seeds, versions, filtering).

Install
  pip install pandas numpy nltk spacy transformers scikit-learn

Run
  python tokenizer_experiments.py ^
      --sentiment training.1600000.processed.noemoticon.csv ^
      --pubmed train.txt --seeds 42 43 44 --detok --out my_results_v2

  (use \\ instead of ^ on Linux/macOS; or put everything on one line)
"""
import argparse
import hashlib
import html
import json
import os
import platform
import random
import re
import sys

import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# =============================================================== data
# Characters that signal a corrupted tweet (U+FFFD or UTF-8 read as cp1252/latin-1).
BAD_CHARS = "\ufffd|Ã|â€|Â|ï¿½"


def _read_csv_robust(path, **kw):
    """Try UTF-8, then cp1252, then latin-1 (which never fails)."""
    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", **kw)


def load_sentiment140(path, drop_bad_chars=True):
    df = _read_csv_robust(path, header=None,
                          names=["target", "id", "date", "flag", "user", "text"],
                          usecols=["target", "text"])
    df["text"] = df["text"].astype(str).map(html.unescape)      # &amp; &quot; &lt; ...
    n_total = len(df)
    n_dropped = 0
    if drop_bad_chars:
        bad = df["text"].str.contains(BAD_CHARS, regex=True)
        n_dropped = int(bad.sum())
        df = df[~bad]
    df = df.reset_index(drop=True)
    df["label"] = (df["target"] == 4).astype(int)
    print(f"Sentiment140: {n_total:,} tweets, dropped {n_dropped:,} with corrupted "
          f"characters, {len(df):,} left")
    out = df[["text", "label"]]
    out.attrs["n_total"] = n_total
    out.attrs["n_dropped_bad_chars"] = n_dropped
    return out


def load_pubmed(path):
    """PubMed RCT train.txt: '###id' header lines, then 'LABEL<TAB>sentence'."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "\t" in line:
                out.append(line.rstrip("\n").split("\t", 1)[1])
    return out


_DETOK = TreebankWordDetokenizer()


def detokenize(s):
    s = s.replace("``", '"').replace("''", '"')
    return _DETOK.detokenize(s.split())


# ======================================================= custom tokenizer
RULES = [  # (name, pattern); order = priority
    ("url", r"https?://\S+|www\.\S+"),
    ("mention", r"@\w+"),
    ("hashtag", r"#\w+"),
    ("emoji", r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"),
    ("abbrev", r"(?:[A-Za-z]\.){2,}"),                       # i.v., i.m.
    ("plusminus", r"\+/-|\+-|±"),                            # -0.03 +/- 0.1
    ("hyphen", r"[A-Za-z0-9]+(?:,[0-9]+)*(?:-[A-Za-z0-9]+)+"),  # 5-fluorouracil, 1,25-dihydroxy..
    ("contraction", r"[A-Za-z]+(?:'[A-Za-z]+)+"),
    ("number", r"(?<!\w)[-+]?\d+(?:[.,]\d+)*"),              # % is its own token
    ("emoticon", r"(?:[:;=][\-o\*']?[\)\]\(\[dDpP/\}\{@\|\\]|<3)"),
    ("repeat_punct", r"[!?.]{2,}"),
    ("word", r"[A-Za-z0-9_]+"),
    ("punct", r"\S"),
]


def make_custom_tokenizer(drop=()):
    parts = [f"(?P<{n}>{p})" for n, p in RULES if n not in drop]
    rx = re.compile("|".join(parts))

    def tok(text):
        out = []
        for m in rx.finditer(text):
            t = m.group(0)
            if m.lastgroup == "url":          # do not swallow trailing punctuation
                t = t.rstrip(".,!?;:)")
            out.append(t)
        return out
    return tok


# ========================================================== tokenizers
def build_tokenizers():
    import spacy
    from transformers import AutoTokenizer

    nlp = spacy.blank("en")                       # tokenizer rules only, no model download
    wp = AutoTokenizer.from_pretrained("bert-base-uncased")
    tweet = TweetTokenizer(preserve_case=True, reduce_len=False)
    word_rx = re.compile(r"\w+|[^\w\s]")

    return {
        "whitespace": lambda s: s.split(),
        "regex": lambda s: word_rx.findall(s),
        "NLTK_word": word_tokenize,
        "TweetTokenizer": tweet.tokenize,
        "spaCy": lambda s: [t.text for t in nlp.make_doc(s) if not t.is_space],
        # tokenize() never adds [CLS]/[SEP]
        "WordPiece": wp.tokenize,
        "CustomTweetRegex": make_custom_tokenizer(),
    }


# ============================================================== metrics
_CHEM_SUFFIX = (r"(?:mab|statin|floxacin|cillin|mycin|azole|prazole|pril|sartan|olol|formin|"
                r"oxetine|tidine|dipine|parin|platin|tinib|navir|fibrate)")
_TIME_WORDS = (r"(?:year|years|week|weeks|month|months|day|days|hour|hours|minute|minutes|"
               r"fold|point|item|old)")

UNIT_PATTERNS = {
    "url": re.compile(r"https?://\S+|www\.\S+"),
    "mention": re.compile(r"@\w+"),
    "hashtag": re.compile(r"#\w+"),
    "emoticon": re.compile(r"(?<!\S)(?:[:;=][\-o\*']?[\)\]\(\[dDpP/\}\{@\|\\]|<3)(?!\S)"),
    "decimal": re.compile(r"(?<!\w)[-+]?\d+\.\d+(?!\w)"),
    "hyphenated": re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b"),
    "abbreviation": re.compile(r"\b(?:[A-Za-z]\.){2,}"),
    "acronym": re.compile(r"\b(?:[A-Z]{2,}(?:-\d+)?|[A-Z]{1,5}-\d+)\b"),
    "chemical": re.compile(
        rf"\b\d+(?:,\d+)*-(?!{_TIME_WORDS}\b)(?:[A-Za-z][A-Za-z\-]{{3,}}|[A-Z]{{2,}})\b"
        rf"|\b[A-Za-z]{{3,}}{_CHEM_SUFFIX}\b", re.I),
    "contraction": re.compile(r"\b[A-Za-z]+'[A-Za-z]+\b"),
    "punct_run": re.compile(r"(?<![!?.])[!?.]{2,}(?![!?.])"),
    "elongation": re.compile(r"(?<![\w/.@#])\w*([A-Za-z])\1{2,}\w*\b"),
    "plusminus": re.compile(r"(?<!\S)(?:\+/-|\+-|±)(?!\S)"),
}


def find_units(text):
    found = []
    for kind, rx in UNIT_PATTERNS.items():
        for m in rx.finditer(text):
            u = m.group(0)
            if kind == "url":
                u = u.rstrip(".,!?;:)")
            found.append((kind, u.lower()))
    return found


def kept_whole(texts, tok_lists):
    """Return {unit_kind: [n_found, n_kept_whole]}."""
    counts = {k: [0, 0] for k in UNIT_PATTERNS}
    for text, toks in zip(texts, tok_lists):
        tokset = {t.lower() for t in toks}
        for kind, unit in find_units(text):
            counts[kind][0] += 1
            counts[kind][1] += int(unit in tokset)
    return counts


def oov_rate(tok_lists, lower=True):
    half = len(tok_lists) // 2
    norm = (lambda t: t.lower()) if lower else (lambda t: t)
    vocab = {norm(t) for toks in tok_lists[:half] for t in toks}
    test = [norm(t) for toks in tok_lists[half:] for t in toks]
    return 100.0 * sum(t not in vocab for t in test) / max(len(test), 1)


def basic_metrics(tok_lists):
    lens = np.array([len(t) for t in tok_lists])
    flat = [t for toks in tok_lists for t in toks]
    return {
        "mean_tokens_doc": lens.mean(),
        "median_tokens_doc": float(np.median(lens)),
        "oov_pct": oov_rate(tok_lists),
        "avg_token_len": float(np.mean([len(t) for t in flat])),
        "total_tokens": int(lens.sum()),
    }


# ========================================================== main runs
def run_main(domains, tokenizers, seeds, out):
    rows, unit_rows, dist = [], [], {}
    for seed in seeds:
        for domain, sampler in domains.items():
            texts = sampler(seed)
            for name, tok in tokenizers.items():
                tl = [tok(t) for t in texts]
                lens = np.minimum(np.array([len(t) for toks in tl for t in toks], dtype=int), 20)
                dist[(domain, name)] = dist.get((domain, name), 0) + np.bincount(lens, minlength=21)
                metrics = basic_metrics(tl)
                metrics["bert_unk_pct"] = (
                    100 * sum(token == "[UNK]" for doc in tl for token in doc)
                    / max(sum(len(doc) for doc in tl), 1)
                    if name == "WordPiece" else np.nan
                )
                rows.append({"seed": seed, "domain": domain, "tokenizer": name, **metrics})
                for kind, (n, k) in kept_whole(texts, tl).items():
                    unit_rows.append({"seed": seed, "domain": domain, "tokenizer": name,
                                      "unit": kind, "found": n, "kept": k})
                print(f"seed={seed} {domain:12s} {name}")

    # token length distribution (share of tokens per length; 20 = "20 or more")
    drows = []
    for (d, n), c in dist.items():
        share = c / max(c.sum(), 1)
        drows += [{"domain": d, "tokenizer": n, "length": L, "share": share[L]} for L in range(1, 21)]
    pd.DataFrame(drows).to_csv(f"{out}/token_length_distribution.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/per_seed_metrics.csv", index=False)
    agg = (df.groupby(["domain", "tokenizer"])
             .agg(["mean", "std"]).drop(columns="seed"))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.round(3).to_csv(f"{out}/metrics_mean_std.csv")

    u = pd.DataFrame(unit_rows)
    u.to_csv(f"{out}/units_per_seed.csv", index=False)
    tot = u.groupby(["domain", "tokenizer", "unit"])[["found", "kept"]].sum().reset_index()
    tot["kept_pct"] = (100 * tot["kept"] / tot["found"].where(tot["found"] > 0)).round(1)
    tot.to_csv(f"{out}/units_kept_whole.csv", index=False)
    return agg, tot


def run_ablation(tweet_sampler, seeds, out):
    variants = {"full": ()} | {f"-{n}": (n,) for n, _ in RULES if n not in ("word", "punct")}
    rows = []
    for seed in seeds:
        texts = tweet_sampler(seed)
        for vname, drop in variants.items():
            tl = [make_custom_tokenizer(drop)(t) for t in texts]
            k = kept_whole(texts, tl)
            found = sum(v[0] for v in k.values())
            kept = sum(v[1] for v in k.values())
            rows.append({"seed": seed, "variant": vname, **basic_metrics(tl),
                         "units_found": found, "units_kept": kept,
                         "kept_pct": 100 * kept / max(found, 1)})
    df = pd.DataFrame(rows)
    res = df.groupby("variant")[["mean_tokens_doc", "oov_pct", "kept_pct"]].agg(["mean", "std"])
    res.round(3).to_csv(f"{out}/ablation_custom_tweet.csv")
    return res


# ========================================================== downstream
def run_downstream(sent_df, tokenizers, seeds, out, n):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from scipy.stats import t as student_t, ttest_rel

    rows = []
    for seed in seeds:
        d = sent_df.sample(n, random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            d["text"], d["label"], test_size=0.2, stratify=d["label"], random_state=seed)
        for name, tok in tokenizers.items():
            vec = TfidfVectorizer(tokenizer=tok, token_pattern=None, lowercase=True,
                                  ngram_range=(1, 2), min_df=2)
            A = vec.fit_transform(Xtr)
            clf = LogisticRegression(max_iter=1000).fit(A, ytr)
            pred = clf.predict(vec.transform(Xte))
            rows.append({"seed": seed, "tokenizer": name,
                         "accuracy": accuracy_score(yte, pred),
                         "macro_f1": f1_score(yte, pred, average="macro"),
                         "n_features": A.shape[1]})
            print(f"downstream seed={seed} {name}: acc={rows[-1]['accuracy']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/downstream_per_seed.csv", index=False)
    res = df.groupby("tokenizer")[["accuracy", "macro_f1", "n_features"]].agg(["mean", "std"])
    res.round(4).to_csv(f"{out}/downstream_sentiment.csv")
    comparisons = []
    for metric in ("accuracy", "macro_f1"):
        scores = df.pivot(index="seed", columns="tokenizer", values=metric)
        names = list(tokenizers)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                differences = (scores[left] - scores[right]).dropna()
                n_pairs = len(differences)
                mean_diff = float(differences.mean()) if n_pairs else np.nan
                if n_pairs >= 2:
                    test = ttest_rel(scores.loc[differences.index, left], scores.loc[differences.index, right])
                    margin = float(student_t.ppf(0.975, n_pairs - 1) * differences.std(ddof=1) / np.sqrt(n_pairs))
                    p_value = float(test.pvalue)
                else:
                    margin = p_value = np.nan
                comparisons.append({
                    "metric": metric, "tokenizer_a": left, "tokenizer_b": right,
                    "n_seed_pairs": n_pairs, "mean_difference_a_minus_b": mean_diff,
                    "ci_95_low": mean_diff - margin if np.isfinite(margin) else np.nan,
                    "ci_95_high": mean_diff + margin if np.isfinite(margin) else np.nan,
                    "p_value": p_value,
                })
    finite = [i for i, row in enumerate(comparisons) if np.isfinite(row["p_value"])]
    ordered = sorted(finite, key=lambda i: comparisons[i]["p_value"])
    adjusted_so_far = 0.0
    for rank, index in enumerate(ordered):
        candidate = (len(ordered) - rank) * comparisons[index]["p_value"]
        adjusted_so_far = max(adjusted_so_far, candidate)
        comparisons[index]["p_value_holm"] = min(adjusted_so_far, 1.0)
    for row in comparisons:
        row.setdefault("p_value_holm", np.nan)
        row["significant_holm_05"] = bool(row["p_value_holm"] < 0.05)
    pd.DataFrame(comparisons).to_csv(f"{out}/downstream_pairwise_tests.csv", index=False)
    return res


# =============================================================== config
def write_config(args, sent, pub, out):
    versions = {"python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__}
    for mod in ("nltk", "spacy", "transformers", "sklearn", "scipy", "altair"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "not installed"
    def sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    cfg = {
        "seeds": args.seeds, "samples_per_domain": args.n, "pubmed_detokenized": args.detok,
        "wordpiece_model": "bert-base-uncased (no [CLS]/[SEP])",
        "oov_protocol": "held-out token rate: vocabulary from first half of each tokenizer sample, measured on second half, lower-cased",
        "bert_unk_protocol": "share of WordPiece output tokens equal to [UNK]; not applicable to other tokenizers",
        "downstream_protocol": {
            "documents_per_seed": args.n_downstream, "test_fraction": 0.2,
            "features": "TF-IDF unigrams and bigrams, min_df=2",
            "classifier": "LogisticRegression(max_iter=1000)",
            "paired_tests": "paired t-tests across seeds; 95% t intervals; Holm correction across all metrics and pairs",
        },
        "source_files": {
            "sentiment140": {"path": os.path.abspath(args.sentiment), "sha256": sha256_file(args.sentiment)},
            "pubmed_rct": {"path": os.path.abspath(args.pubmed), "sha256": sha256_file(args.pubmed)},
        },
        "code_sha256": sha256_file(__file__),
        "sentiment140_total": sent.attrs.get("n_total"),
        "sentiment140_dropped_corrupted": sent.attrs.get("n_dropped_bad_chars"),
        "sentiment140_available": len(sent), "pubmed_available": len(pub),
        "versions": versions, "command": " ".join(sys.argv),
    }
    with open(f"{out}/run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ================================================================== cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment", required=True)
    ap.add_argument("--pubmed", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--n", type=int, default=5000, help="samples per domain")
    ap.add_argument("--n_downstream", type=int, default=20000)
    ap.add_argument("--detok", action="store_true", help="detokenize PubMed text")
    ap.add_argument("--keep_bad_chars", action="store_true",
                    help="do not drop tweets with corrupted characters")
    ap.add_argument("--out", default="my_results_v2")
    ap.add_argument("--skip_downstream", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sent = load_sentiment140(args.sentiment, drop_bad_chars=not args.keep_bad_chars)
    pub = load_pubmed(args.pubmed)
    print(f"PubMed RCT: {len(pub):,} sentences")

    def tweets(seed):
        return sent.sample(args.n, random_state=seed)["text"].tolist()

    def pubmed(seed):
        s = random.Random(seed).sample(pub, args.n)
        return [detokenize(x) for x in s] if args.detok else s

    domains = {"Sentiment140": tweets, "PubMed RCT": pubmed}
    toks = build_tokenizers()

    agg, units = run_main(domains, toks, args.seeds, args.out)
    print("\n=== mean +- std over seeds ===\n", agg.round(3))
    print("\n=== units kept whole ===\n", units.to_string(index=False))

    print("\n=== CustomTweetRegex ablation ===\n",
          run_ablation(tweets, args.seeds, args.out).round(3))

    if not args.skip_downstream:
        print("\n=== downstream sentiment ===\n",
              run_downstream(sent, toks, args.seeds, args.out, args.n_downstream))

    write_config(args, sent, pub, args.out)
    print(f"\nCSV files saved in {args.out}/")


if __name__ == "__main__":
    main()