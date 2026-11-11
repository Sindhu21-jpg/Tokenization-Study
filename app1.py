# dashboard.py  --  run with:  python -m streamlit run dashboard.py
import html
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Text & Tokenization Analysis",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
SUMMARY_NAME = "summary_metrics.csv"
LENGTH_NAME = "token_length_distribution.csv"
MANUAL_NAMES = ["manual_inspection_30.csv", "manual_inspection_examples.csv"]

TOKENIZER_ORDER = [
    "whitespace", "regex", "NLTK_word", "TweetTokenizer",
    "spaCy", "WordPiece", "CustomTweetRegex",
]
REQUIRED_COLS = [
    "domain", "tokenizer", "documents", "mean_tokens_per_document",
    "median_tokens_per_document", "mean_token_length_chars",
    "token_length_p50_chars", "token_length_p90_chars",
    "oov_rate_tokens", "oov_rate_types", "unk_rate", "total_tokens",
]

# Phenomena used to check whether a tokenizer keeps a unit as ONE token.
PRESERVE_CHECKS = {
    "URL": re.compile(r"(?:https?://|www\.)\S*[^\s.,;:!?)\]\"']"),
    "mention": re.compile(r"(?<![\w@])@\w+"),
    "hashtag": re.compile(r"(?<![\w#])#\w+"),
    "emoticon": re.compile(
        r"(?<!\w)(?:<3|[:;=8xX][\-^']?[)(/\\DPpOo|]|[)(/\\DPpOo|][\-^']?[:;=8xX])(?!\w)"
    ),
    "contraction": re.compile(r"\b\w+['’]\w+\b"),
    "decimal number": re.compile(r"\b\d+[.,]\d+\b"),
    "hyphenated term": re.compile(r"\b\w+(?:-\w+)+\b"),
    "abbreviation": re.compile(r"\b[A-Z]{2,}s?\b"),
}

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #888; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .section-title { font-size: 1.5rem; font-weight: 650; margin-top: 1.2rem; margin-bottom: 0.8rem; }
    .token-box {
        padding: 6px 9px; margin: 3px; border-radius: 8px; border: 1px solid #d9d9d9;
        display: inline-block; background: #f3f3f3; color: #222;
        font-family: monospace; font-size: 0.9rem;
    }
    .small-note { color: #888; font-size: 0.9rem; }
    div[data-testid="stMetric"] { border: 1px solid #e0e0e0; padding: 12px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def _read_csv(path_str, mtime):
    # mtime is part of the cache key, so edited files are re-read.
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path_str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path_str, encoding="latin-1")


def load_csv(path):
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    try:
        return _read_csv(str(p), p.stat().st_mtime)
    except Exception:
        return pd.DataFrame()


def show_df(df, **kwargs):
    """Full-width dataframe that works on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def order_tokenizers(names):
    names = list(names)
    return [t for t in TOKENIZER_ORDER if t in names] + [t for t in names if t not in TOKENIZER_ORDER]


def grouped_bar(df, value_col, title, ylabel, scale=1.0, decimals=1):
    """One group per tokenizer, one bar per dataset."""
    pivot = df.pivot(index="tokenizer", columns="domain", values=value_col) * scale
    pivot = pivot.reindex(order_tokenizers(pivot.index))
    n = max(len(pivot.columns), 1)
    width = 0.8 / n
    x = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, dom in enumerate(pivot.columns):
        values = pivot[dom].values
        bars = ax.bar(x + (k - (n - 1) / 2) * width, values, width, label=dom)
        ax.bar_label(bars, labels=[f"{v:.{decimals}f}" for v in values], fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.12)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def tokens_to_html(tokens):
    if not tokens:
        return "<span class='small-note'>No tokens generated.</span>"
    return "".join(f"<span class='token-box'>{html.escape(str(t))}</span>" for t in tokens)


def split_tokens(value):
    return [] if pd.isna(value) else str(value).split(" | ")


def text_statistics(tokens):
    if not tokens:
        return {"Token Count": 0, "Unique Tokens": 0, "UNK Tokens": 0,
                "Average Token Length": 0.0, "Median Token Length": 0.0,
                "Minimum Length": 0, "Maximum Length": 0}
    lengths = [len(t) for t in tokens]
    return {
        "Token Count": len(tokens),
        "Unique Tokens": len(set(tokens)),
        "UNK Tokens": tokens.count("[UNK]"),
        "Average Token Length": round(float(np.mean(lengths)), 2),
        "Median Token Length": round(float(np.median(lengths)), 2),
        "Minimum Length": min(lengths),
        "Maximum Length": max(lengths),
    }


def preservation_table(texts, tokens_per_text):
    """% of hashtags, URLs, ... that each tokenizer keeps as exactly one token.
    tokens_per_text[i] is a dict {tokenizer name: token list} for texts[i]."""
    totals = Counter()
    intact = defaultdict(Counter)
    names = []
    for text, by_name in zip(texts, tokens_per_text):
        for name in by_name:
            if name not in names:
                names.append(name)
        for cat, pat in PRESERVE_CHECKS.items():
            matches = [m.group(0).casefold() for m in pat.finditer(str(text))]
            if not matches:
                continue
            totals[cat] += len(matches)
            for name, toks in by_name.items():
                token_set = {t.casefold() for t in toks}
                intact[name][cat] += sum(m in token_set for m in matches)
    if not totals:
        return pd.DataFrame()
    rows = []
    for name in order_tokenizers(names):
        row = {"Tokenizer": name}
        for cat, total in totals.items():
            row[f"{cat} (n={total})"] = round(100 * intact[name][cat] / total, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_findings(df):
    out = []
    for dom, g in df.groupby("domain"):
        most = g.loc[g["mean_tokens_per_document"].idxmax()]
        fewest = g.loc[g["mean_tokens_per_document"].idxmin()]
        low = g.loc[g["oov_rate_tokens"].idxmin()]
        high = g.loc[g["oov_rate_tokens"].idxmax()]
        out.append(
            f"**{dom}:** {most['tokenizer']} produces the most tokens per document "
            f"({most['mean_tokens_per_document']:.1f}) and {fewest['tokenizer']} the fewest "
            f"({fewest['mean_tokens_per_document']:.1f}). {low['tokenizer']} has the lowest OOV "
            f"rate ({low['oov_rate_tokens'] * 100:.2f}%) and {high['tokenizer']} the highest "
            f"({high['oov_rate_tokens'] * 100:.2f}%)."
        )

    pivot = df.pivot(index="tokenizer", columns="domain", values="mean_tokens_per_document")
    if {"Sentiment140", "PubMed RCT"} <= set(pivot.columns):
        ratio = (pivot["PubMed RCT"] / pivot["Sentiment140"]).dropna().mean()
        if pd.notna(ratio):
            out.append(
                f"PubMed RCT sentences yield on average **{ratio:.1f}×** more tokens than tweets "
                f"across the selected tokenizers, because they are longer and contain technical terms."
            )

    wp = df[df["tokenizer"].str.lower() == "wordpiece"]
    for _, r in wp.iterrows():
        out.append(
            f"**WordPiece** on {r['domain']}: OOV {r['oov_rate_tokens'] * 100:.2f}%, "
            f"[UNK] {r['unk_rate'] * 100:.3f}%, {r['mean_tokens_per_document']:.1f} tokens/doc. "
            f"Its OOV is low by design (unseen words are split into sub-word pieces), "
            f"at the cost of more tokens per document."
        )

    sent = df[df["domain"] == "Sentiment140"].set_index("tokenizer")
    if "regex" in sent.index:
        base = sent.loc["regex", "oov_rate_tokens"] * 100
        parts = [
            f"{n}: {sent.loc[n, 'oov_rate_tokens'] * 100:.2f}%"
            for n in ("TweetTokenizer", "CustomTweetRegex") if n in sent.index
        ]
        if parts:
            out.append(
                f"On Sentiment140, OOV is {', '.join(parts)} versus {base:.2f}% for the plain regex "
                f"baseline. Tweet-aware tokenizers keep hashtags, mentions and URLs whole; such "
                f"strings are mostly unique, which raises token-level OOV. OOV alone therefore does "
                f"not measure tokenization quality, so use the manual inspection table to judge it."
            )
    return out


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🔤 Text Analysis")
page = st.sidebar.radio("Select Mode", ["📊 Results Dashboard", "🔍 Analyze Your Own Text"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project**

    Comparative Study of Tokenization Methods on Social Media and Biomedical Text

    **Datasets**
    - Sentiment140
    - PubMed RCT

    **Tokenizers**
    - Whitespace
    - Regex
    - NLTK Word
    - TweetTokenizer
    - spaCy
    - WordPiece
    - CustomTweetRegex
    """
)
results_dir_text = st.sidebar.text_input("Results folder", value="my_results")
results_dir = Path(results_dir_text)
if not results_dir.is_absolute():
    results_dir = BASE_DIR / results_dir

# ============================================================
# PAGE 1: RESULTS DASHBOARD
# ============================================================
if page == "📊 Results Dashboard":
    st.markdown("<div class='main-title'>🔤 Text & Tokenization Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Comparative analysis of seven tokenization methods "
        "on social media and biomedical text.</div>",
        unsafe_allow_html=True,
    )

    summary = load_csv(results_dir / SUMMARY_NAME)
    if summary.empty:
        st.error(
            f"{SUMMARY_NAME} was not found in {results_dir}. Run "
            f"`python tokenization_study.py --sentiment140 ... --pubmed-rct ... --out {results_dir_text}` first."
        )
        st.stop()
    missing = [c for c in REQUIRED_COLS if c not in summary.columns]
    if missing:
        st.error(
            f"{SUMMARY_NAME} is missing columns {missing}. It was probably produced by an older "
            f"version of the script. Re-run the corrected tokenization_study.py."
        )
        st.stop()

    length_df = load_csv(results_dir / LENGTH_NAME)
    manual = pd.DataFrame()
    for name in MANUAL_NAMES:
        manual = load_csv(results_dir / name)
        if not manual.empty:
            break

    # ---- filters
    st.sidebar.subheader("Dashboard Filters")
    domains = list(summary["domain"].dropna().unique())
    selected_domains = st.sidebar.multiselect("Dataset / Domain", domains, default=domains)
    tokenizers_all = order_tokenizers(summary["tokenizer"].dropna().unique())
    selected_tokenizers = st.sidebar.multiselect("Tokenizer", tokenizers_all, default=tokenizers_all)

    filtered = summary[
        summary["domain"].isin(selected_domains) & summary["tokenizer"].isin(selected_tokenizers)
    ].copy()

    # ---- overview
    section("📌 Study Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", summary["domain"].nunique())
    c2.metric("Tokenizers", summary["tokenizer"].nunique())
    c3.metric("Samples / Dataset", f"{int(summary['documents'].max()):,}")
    c4.metric("Tokens Analysed", f"{int(summary['total_tokens'].sum()):,}")

    section("📚 Dataset Information")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """
            ### 🐦 Sentiment140

            Social-media text containing short, informal messages.

            **Characteristics**
            - Informal language
            - Hashtags
            - Mentions
            - URLs
            - Abbreviations
            - Misspellings
            - Repeated letters and punctuation
            """
        )
        st.caption(
            "The standard training file is the 'noemoticon' version, so emoticons were removed "
            "from the data and appear only rarely."
        )
    with d2:
        st.markdown(
            """
            ### 🧬 PubMed RCT

            Biomedical research sentences extracted from randomized controlled trial abstracts.

            **Characteristics**
            - Scientific terminology
            - Long sentences
            - Chemical and drug names
            - Numbers and statistics
            - Abbreviations
            - Technical vocabulary
            """
        )

    with st.expander("Methodology and metric definitions"):
        st.markdown(
            """
            - Each dataset is a random sample (fixed seed) of **5,000** documents.
            - **Tokens per document** and **token length** use all sampled documents.
            - **OOV rate**: for each tokenizer, the vocabulary is built from the **first half** of the
              sample (lower-cased tokens), and OOV is measured on the **second half**. *Tokens* is the share
              of running tokens not in the vocabulary; *types* is the share of distinct tokens not in it.
            - **UNK rate**: share of `[UNK]` tokens (relevant for WordPiece, e.g. emoji).
            - WordPiece low OOV is a property of sub-word tokenization, so compare it with word-level
              tokenizers carefully.
            """
        )

    # ---- performance table
    section("📊 Tokenizer Performance")
    if filtered.empty:
        st.warning("No data matches the selected filters.")
    else:
        cols = [
            "domain", "tokenizer", "documents", "mean_tokens_per_document",
            "median_tokens_per_document", "mean_token_length_chars",
            "token_length_p50_chars", "token_length_p90_chars",
            "oov_rate_tokens", "oov_rate_types", "unk_rate",
        ]
        table = filtered[cols].copy()
        for c in ("oov_rate_tokens", "oov_rate_types"):
            table[c] = (table[c] * 100).round(2)
        table["unk_rate"] = (table["unk_rate"] * 100).round(4)
        table["mean_tokens_per_document"] = table["mean_tokens_per_document"].round(2)
        table["mean_token_length_chars"] = table["mean_token_length_chars"].round(2)
        table = table.rename(columns={
            "domain": "Domain", "tokenizer": "Tokenizer", "documents": "Documents",
            "mean_tokens_per_document": "Mean Tokens/Doc",
            "median_tokens_per_document": "Median Tokens/Doc",
            "mean_token_length_chars": "Mean Token Length",
            "token_length_p50_chars": "P50 Length", "token_length_p90_chars": "P90 Length",
            "oov_rate_tokens": "OOV Tokens (%)", "oov_rate_types": "OOV Types (%)",
            "unk_rate": "UNK Rate (%)",
        })
        show_df(table)
        st.download_button("⬇️ Download Performance Table", csv_bytes(table),
                           "tokenizer_performance.csv", "text/csv")

        # ---- charts
        section("📈 Mean Tokens per Document")
        grouped_bar(filtered, "mean_tokens_per_document",
                    "Average Number of Tokens Generated", "Mean tokens per document")

        section("📉 OOV Rate")
        grouped_bar(filtered, "oov_rate_tokens", "Out-of-Vocabulary Token Rate",
                    "OOV rate (%)", scale=100, decimals=2)
        st.caption("Vocabulary built from the first half of each sample by the same tokenizer; "
                   "measured on the second half.")

        section("📏 Average Token Length")
        grouped_bar(filtered, "mean_token_length_chars", "Average Token Length",
                    "Characters", decimals=2)

        # ---- length distribution
        section("📊 Token Length Distribution")
        if length_df.empty or not {"domain", "tokenizer", "token_length", "proportion"} <= set(length_df.columns):
            st.info(f"{LENGTH_NAME} was not found in {results_dir}. Re-run the corrected "
                    f"tokenization_study.py to generate it.")
        else:
            dist_domain = st.selectbox("Dataset for distribution", selected_domains or domains)
            dist = length_df[
                (length_df["domain"] == dist_domain)
                & length_df["tokenizer"].isin(selected_tokenizers)
            ].pivot(index="token_length", columns="tokenizer", values="proportion")
            dist = dist[order_tokenizers(dist.columns)]
            st.line_chart(dist)
            st.caption("Share of tokens by length in characters. Lengths of 20 or more are grouped in the last bin.")

        # ---- cross-domain comparison
        section("🔬 Domain Comparison")
        tmp = filtered.assign(oov_pct=filtered["oov_rate_tokens"] * 100)
        cmp = tmp.pivot(index="tokenizer", columns="domain",
                        values=["mean_tokens_per_document", "oov_pct"])
        cmp.columns = [
            f"{'Tokens/Doc' if m == 'mean_tokens_per_document' else 'OOV tokens (%)'}: {d}"
            for m, d in cmp.columns
        ]
        cmp = cmp.reindex(order_tokenizers(cmp.index)).round(2).reset_index()
        cmp = cmp.rename(columns={"tokenizer": "Tokenizer"})
        show_df(cmp)

        # ---- findings (computed from the data)
        section("💡 Key Findings")
        for finding in build_findings(filtered):
            st.markdown(f"- {finding}")

    # ---- manual inspection
    section("🔎 Manual Inspection Examples")
    if manual.empty:
        st.info(
            f"No inspection file found in {results_dir}. Expected one of: {', '.join(MANUAL_NAMES)}. "
            f"It is produced by tokenization_study.py."
        )
    else:
        manual = manual.copy()
        if "manual_error_notes" in manual.columns:
            manual["manual_error_notes"] = manual["manual_error_notes"].fillna("").astype(str)
            filled = int(manual["manual_error_notes"].str.strip().ne("").sum())
            st.caption(f"{filled} of {len(manual)} examples have notes in 'manual_error_notes'. "
                       f"Edit the CSV to add your observations, then refresh this page.")

        shown = manual
        if "domain" in manual.columns:
            pick = st.selectbox("Examples from", ["Both"] + list(manual["domain"].dropna().unique()))
            if pick != "Both":
                shown = manual[manual["domain"] == pick]
        show_df(shown)
        st.download_button("⬇️ Download Manual Inspection", csv_bytes(manual),
                           "manual_inspection_30.csv", "text/csv")

        tok_cols = [t for t in order_tokenizers(summary["tokenizer"].unique()) if t in manual.columns]
        if tok_cols and "text" in manual.columns:
            pres = preservation_table(
                manual["text"].tolist(),
                [{t: split_tokens(row[t]) for t in tok_cols} for _, row in manual.iterrows()],
            )
            if not pres.empty:
                st.markdown("#### Share of units kept as a single token (inspected examples)")
                show_df(pres)
                st.caption("Computed on the inspected examples only, so it is not representative "
                           "of the whole corpus. WordPiece lower-cases and splits into sub-words, "
                           "so it rarely keeps a unit whole by design.")

# ============================================================
# PAGE 2: ANALYZE YOUR OWN TEXT
# ============================================================
else:
    st.markdown("<div class='main-title'>🔍 Analyze Your Own Text</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Enter your own social-media or biomedical text and compare "
        "how all seven tokenizers process it.</div>",
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="Loading tokenizers (first run downloads NLTK data and the BERT vocabulary)...")
    def load_tokenizers():
        from tokenization_study import get_tokenizers   # same tokenizers as the study
        return get_tokenizers()

    section("1️⃣ Select Text Domain")
    domain = st.radio(
        "Choose the type of text:",
        ["🐦 Social Media / Sentiment140", "🧬 Biomedical / PubMed RCT"],
        horizontal=True,
    )
    is_social = "Social Media" in domain

    if is_social:
        example_text = ("I absolutely LOVE this product!!! 😍 Can't wait :) "
                        "Check https://example.com #AmazingProduct @brand "
                        "but the delivery was sooo slow!!!")
    else:
        example_text = ("Patients received 5-fluorouracil (5-FU) 400 mg/m2 and IL-6 levels fell "
                        "by 12.5% in type 2 diabetes mellitus (p < 0.05).")

    section("2️⃣ Enter Text")
    st.session_state.setdefault("user_text", "")

    def set_example(text):
        st.session_state["user_text"] = text

    st.button("Load example text", on_click=set_example, args=(example_text,))
    user_text = st.text_area(
        "Enter a sentence, paragraph, tweet, abstract sentence, or other text:",
        height=160,
        placeholder="Type or paste your text here...",
        key="user_text",
    )

    if st.button("🔍 Analyze Text", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text before clicking Analyze.")
        else:
            try:
                tokenizers = load_tokenizers()
            except Exception as e:
                st.error(
                    f"Could not load the tokenizers: {e}. Make sure tokenization_study.py is in the "
                    f"same folder as this file, the packages are installed, and you are online "
                    f"for the first run."
                )
                st.stop()
            results, errors = {}, []
            for name, fn in tokenizers.items():
                try:
                    results[name] = [str(t) for t in (fn(user_text) or [])]
                except Exception as e:
                    results[name] = []
                    errors.append(f"{name} could not process the text: {e}")
            # Stored in session_state so results stay visible when widgets below are used.
            st.session_state["analysis"] = {
                "text": user_text, "domain": domain, "tokens": results, "errors": errors,
            }

    analysis = st.session_state.get("analysis")
    if analysis:
        text, results, social = analysis["text"], analysis["tokens"], "Social Media" in analysis["domain"]
        for msg in analysis["errors"]:
            st.warning(msg)

        st.markdown("---")
        section("📄 Input Text")
        st.info(text)

        section("📌 Input Statistics")
        a, b, c = st.columns(3)
        a.metric("Characters", len(text))
        b.metric("Whitespace Words", len(text.split()))
        c.metric("Estimated Sentences", max(1, len(re.findall(r"[.!?]+(?:\s|$)", text))))

        stats_df = pd.DataFrame(
            [{"Tokenizer": n, **text_statistics(t)} for n, t in results.items()]
        )

        section("📊 Tokenizer Comparison")
        show_df(stats_df)
        st.download_button("⬇️ Download Tokenization Statistics", csv_bytes(stats_df),
                           "manual_tokenization_results.csv", "text/csv")

        section("📈 Token Count Comparison")
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(stats_df["Tokenizer"], stats_df["Token Count"])
        ax.bar_label(bars, padding=2)
        ax.set_ylabel("Number of Tokens")
        ax.set_title("Tokens Generated by Each Tokenizer")
        ax.margins(y=0.12)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        pres = preservation_table([text], [results])
        if not pres.empty:
            section("🧩 Are Special Units Kept Whole?")
            show_df(pres)
            st.caption("For each URL, mention, hashtag, emoticon, contraction, decimal, hyphenated term or "
                       "abbreviation found in your text: the % that the tokenizer outputs as exactly one "
                       "token (case-insensitive). WordPiece splits into sub-words by design.")

        section("🔤 Detailed Tokenization Output")
        tabs = st.tabs(list(results.keys()))
        for tab, (name, toks) in zip(tabs, results.items()):
            with tab:
                st.write(f"**{len(toks)} tokens generated**")
                st.markdown(tokens_to_html(toks), unsafe_allow_html=True)
                st.code(" | ".join(toks) if toks else "(none)", language="text")

        long_df = pd.DataFrame(
            [{"Tokenizer": n, "Position": i + 1, "Token": t, "Length": len(t)}
             for n, toks in results.items() for i, t in enumerate(toks)]
        )
        if not long_df.empty:
            st.download_button("⬇️ Download All Token Outputs", csv_bytes(long_df),
                               "all_tokenizer_outputs.csv", "text/csv")

        section("💡 Interpretation")
        counts = stats_df.set_index("Tokenizer")["Token Count"]
        st.markdown(f"- **{counts.idxmax()}** produced the most tokens (**{counts.max()}**).")
        st.markdown(f"- **{counts.idxmin()}** produced the fewest tokens (**{counts.min()}**).")
        unk = int(stats_df.loc[stats_df["Tokenizer"] == "WordPiece", "UNK Tokens"].sum()) \
            if "WordPiece" in results else 0
        if unk:
            st.markdown(f"- WordPiece produced **{unk} [UNK]** token(s): characters missing from "
                        f"the BERT vocabulary, such as emoji.")
        if social:
            st.markdown(
                "- For social-media text, check hashtags, mentions, URLs, emoticons, elongated words and "
                "repeated punctuation. Tweet-aware tokenizers (TweetTokenizer, CustomTweetRegex) are "
                "designed to keep these whole; see the table above."
            )
        else:
            st.markdown(
                "- For biomedical text, check hyphenated chemical names, decimals, units, parentheses and "
                "abbreviations. Generic tokenizers often split these; WordPiece splits rare terms into "
                "sub-word pieces instead of producing an unknown word."
            )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Comparative Study of Tokenization Methods on Social Media and Biomedical Text | M.Tech Project")