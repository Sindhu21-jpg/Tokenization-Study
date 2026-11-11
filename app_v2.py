"""
Streamlit front end for the tokenization study (v2 results).

Needs tokenizer_experiments.py in the same folder and the files it writes
(default folder: my_results_v2).

Run:
  python -m streamlit run app_v2.py --server.fileWatcherType none
"""
import json
import os
import re

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tokenizer_experiments import build_tokenizers, kept_whole

st.set_page_config(page_title="Tokenization Studio", page_icon="🔤", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700&display=swap');
:root {
    --ink: #183b39;
    --muted: #657673;
    --paper: #f3f7f5;
    --surface: #ffffff;
    --line: #dce6e1;
    --accent: #d9684e;
}
html, body, [class*="css"] { font-family: "DM Sans", sans-serif; letter-spacing: 0; }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #183b39; border-right: 1px solid #315855; }
[data-testid="stSidebar"] * { color: #f3f7f5; }
[data-testid="stSidebar"] input { color: #ff4b4b !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #b5c8c2; }
h1, h2, h3 { color: var(--ink); font-family: "Manrope", "DM Sans", sans-serif; letter-spacing: 0; }
h1 { font-size: 2.35rem; font-weight: 700; }
h2 { font-size: 1.45rem; font-weight: 700; }
h3 { font-size: 1.05rem; font-weight: 700; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 5px; padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; padding: 0.8rem 1rem; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink); border-bottom: 3px solid var(--accent); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 5px; }
button[kind="primary"] { background: var(--ink); border-color: var(--ink); }
button[kind="primary"]:hover { background: #285b55; border-color: #285b55; }
</style>
""", unsafe_allow_html=True)
APP_BUILD = "2026-10-14 d"   # shown in the sidebar so you can see which file is running

ORDER = ["whitespace", "regex", "NLTK_word", "TweetTokenizer", "spaCy", "WordPiece", "CustomTweetRegex"]
EXAMPLES = {
    "🐦 Social Media / Sentiment140": (
        "I absolutely LOVE this product!!! 😍 Can't wait :) Check https://example.com "
        "#AmazingProduct @brand but the delivery was sooo slow!!!"),
    "🧬 Biomedical / PubMed RCT": (
        "Both fenofibrate and atorvastatin significantly decreased apo CIII levels by -0.03 +/- 0.01 mmol/L "
        "(p < 0.05), and i.v. administered danofloxacin was well-tolerated in H. pylori-positive patients "
        "treated with 5-fluorouracil (5-FU)."),
}


# ================================================================ helpers
@st.cache_resource(show_spinner="Loading tokenizers (first time only)...")
def get_tokenizers():
    return build_tokenizers()


def _flat(df):
    df.columns = ["_".join(str(x) for x in c if "Unnamed" not in str(x)) if isinstance(c, tuple) else c
                  for c in df.columns]
    return df


def load(folder, name, multi_header=False, index_cols=0):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    try:
        if multi_header:
            return _flat(pd.read_csv(path, header=[0, 1], index_col=0)).reset_index()
        return pd.read_csv(path, index_col=index_cols, encoding="utf-8-sig").reset_index()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read {name}: {e}")
        return None


def mean_std(df, base, nd=2):
    m, s = df[f"{base}_mean"], df.get(f"{base}_std")
    if s is None:
        return m.round(nd).astype(str)
    return m.round(nd).astype(str) + " ± " + s.round(nd).fillna(0).astype(str)


def grouped_bar(df, base, title, ylabel, decimals=1):
    """One group per tokenizer, one bar per dataset (side by side, with std error bars)."""
    st.subheader(title)
    mean = df.pivot(index="tokenizer", columns="domain", values=f"{base}_mean")
    mean = mean.reindex([t for t in ORDER if t in mean.index])
    if f"{base}_std" in df:
        err = df.pivot(index="tokenizer", columns="domain", values=f"{base}_std").reindex(mean.index).fillna(0)
    else:
        err = mean * 0
    n = max(len(mean.columns), 1)
    width = 0.8 / n
    x = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, dom in enumerate(mean.columns):
        vals = mean[dom].values
        bars = ax.bar(x + (k - (n - 1) / 2) * width, vals, width, label=dom,
                      yerr=err[dom].values, capsize=2, error_kw={"elinewidth": 0.8})
        ax.bar_label(bars, labels=[f"{v:.{decimals}f}" for v in vals], fontsize=7, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(mean.index, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.14)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def estimate_sentences(text):
    """Rough sentence count: ignores URLs, decimals and dotted abbreviations (i.v., H.)."""
    t = re.sub(r"https?://\S+|www\.\S+", " ", text)
    t = re.sub(r"\b(?:[A-Za-z]\.)+", " ", t)
    return max(1, len(re.findall(r"[.!?]+(?=\s|$)", t.strip())))


def download(df, label, fname, key=None):
    st.download_button(f"⬇️ {label}", df.to_csv(index=False).encode("utf-8"), fname, "text/csv", key=key)


def styled(df, fmt="{:.1f}"):
    try:
        return df.style.format(fmt, na_rep="–").background_gradient(cmap="RdYlGn", vmin=0, vmax=100)
    except Exception:  # noqa: BLE001  (matplotlib/jinja problem -> plain table)
        return df


# ================================================================ sidebar
st.sidebar.title("Tokenization Lab")
st.sidebar.caption("SOCIAL MEDIA  /  BIOMEDICAL TEXT")
mode = st.sidebar.radio("Select mode", ["📊 Results Dashboard", "🔍 Analyze Your Own Text"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Study workspace**")
folder = st.sidebar.text_input("Results folder", "my_results_v2")
st.sidebar.caption(f"app_v2 · build {APP_BUILD}")


# ============================================================== sections
def dataset_info():
    st.header("📚 Dataset Information")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🐦 Sentiment140")
        st.write("Social-media text containing short, informal messages.")
        st.markdown("**Characteristics**")
        st.markdown("- Informal language\n- Hashtags\n- Mentions\n- URLs\n- Abbreviations\n"
                    "- Misspellings\n- Repeated letters and punctuation")
        st.caption("The standard training file is the 'noemoticon' version, so emoticons were removed "
                   "from the data and appear only rarely.")
    with c2:
        st.subheader("🧬 PubMed RCT")
        st.write("Biomedical research sentences extracted from randomized controlled trial abstracts.")
        st.markdown("**Characteristics**")
        st.markdown("- Scientific terminology\n- Long sentences\n- Chemical and drug names\n"
                    "- Numbers and statistics\n- Abbreviations\n- Technical vocabulary")
    with st.expander("Methodology and metric definitions"):
        st.markdown(
            "- **Sampling:** random samples with fixed seeds; results are mean ± std over seeds. "
            "Tweets with corrupted characters are removed first; PubMed text can be detokenized "
            "(see `run_config.json`).\n"
            "- **Tokens per document:** number of tokens a tokenizer outputs for one text "
            "(WordPiece without `[CLS]`/`[SEP]`).\n"
            "- **Held-out OOV:** vocabulary built from the first half of each tokenizer's sample; percentage "
            "of second-half tokens not in that vocabulary. It is tokenizer-specific and not directly comparable.\n"
            "- **BERT [UNK]:** percentage of WordPiece outputs that equal `[UNK]`; reported separately and "
            "not applicable to the other tokenizers.\n"
            "- **Average token length / distribution:** characters per token; lengths of 20 or more share "
            "the last bin.\n"
            "- **Units kept whole:** % of URLs, mentions, hashtags, emoticons, decimals, hyphenated terms, "
            "dotted abbreviations, acronyms, chemical/drug names, contractions, punctuation runs, "
            "elongated words and ± signs that the tokenizer outputs as exactly one token (case-insensitive). "
            "Chemical names are detected with a pattern (digit-hyphen names such as 5-fluorouracil and drug "
            "suffixes such as -statin, -floxacin, -mab), so it is an approximation.\n"
            "- **Downstream tests:** paired t-tests compare same-seed Sentiment140 scores; intervals use the t "
            "distribution and p-values are Holm-adjusted across tokenizer pairs and metrics.\n"
            "- **Manual review:** automatic flags/suggestions are screening aids, not annotations. Agreement and "
            "Cohen's κ are shown only after reviewers enter labels.\n"
            "- **WordPiece:** `bert-base-uncased`; it splits into sub-words by design, so it keeps few units whole.")


def length_distribution(toks):
    st.header("📊 Token length distribution")
    dist = load(folder, "token_length_distribution.csv", index_cols=None)
    if dist is None or "share" not in dist:
        st.info("token_length_distribution.csv not found in this folder. Re-run tokenizer_experiments.py "
                "(the latest version saves it).")
        return
    dom = st.selectbox("Dataset for distribution", sorted(dist["domain"].unique()), key="dist_dom")
    d = dist[(dist["domain"] == dom) & dist["tokenizer"].isin(toks)]
    if d.empty:
        st.info("Select at least one tokenizer to display its length distribution.")
        return
    d = d.copy()
    d["length_label"] = d["length"].map(lambda length: "20+" if length >= 20 else str(int(length)))
    length_order = [str(length) for length in range(1, 20)] + ["20+"]
    tokenizer_order = [tokenizer for tokenizer in ORDER if tokenizer in d["tokenizer"].unique()]
    chart = alt.Chart(d).mark_rect(stroke="#ffffff", strokeWidth=1).encode(
        x=alt.X("length_label:N", sort=length_order, title="Token length (characters)",
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        y=alt.Y("tokenizer:N", sort=tokenizer_order, title=None),
        color=alt.Color("share:Q", title="Share of tokens",
                        scale=alt.Scale(scheme="viridis", domainMin=0, clamp=True),
                        legend=alt.Legend(format=".0%")),
        tooltip=[alt.Tooltip("tokenizer:N", title="Tokenizer"),
                 alt.Tooltip("length_label:N", title="Length"),
                 alt.Tooltip("share:Q", title="Share", format=".2%")],
    ).properties(height=max(260, 36 * len(tokenizer_order)))
    st.altair_chart(chart, width="stretch")
    st.caption("Each row is normalized to 100%. Color indicates share; hover for exact percentages. Lengths of 20 or more share the final bin.")


def domain_comparison(m):
    st.header("🔬 Domain comparison")
    order = [t for t in ORDER if t in m["tokenizer"].values]
    t = m.pivot(index="tokenizer", columns="domain", values="mean_tokens_doc_mean").reindex(order)
    o = m.pivot(index="tokenizer", columns="domain", values="oov_pct_mean").reindex(order)
    out = pd.concat([t.add_prefix("Tokens/Doc: "), o.add_prefix("OOV %: ")], axis=1)
    if {"PubMed RCT", "Sentiment140"} <= set(t.columns):
        out["PubMed / tweet tokens"] = t["PubMed RCT"] / t["Sentiment140"]
    st.dataframe(out.round(2))


def key_findings(m, units, domains):
    st.header("💡 Key Findings")
    for dom in domains:
        d = m[m["domain"] == dom].set_index("tokenizer")
        if d.empty:
            continue
        t, o = d["mean_tokens_doc_mean"], d["oov_pct_mean"]
        st.markdown(
            f"- **{dom}:** {t.idxmax()} produces the most tokens per document ({t.max():.1f}) and "
            f"{t.idxmin()} the fewest ({t.min():.1f}). {o.idxmin()} has the lowest OOV ({o.min():.2f}%) "
            f"and {o.idxmax()} the highest ({o.max():.2f}%).")
        if units is not None:
            u = units[(units["domain"] == dom) & (units["found"] >= 10) & units["tokenizer"].isin(d.index)]
            if not u.empty:
                avg = u.groupby("tokenizer")["kept_pct"].mean()
                st.markdown(
                    f"  - Units kept whole (average over unit types with at least 10 occurrences): "
                    f"{avg.idxmax()} is best ({avg.max():.0f}%) and {avg.idxmin()} is lowest ({avg.min():.0f}%).")
    st.markdown("- Held-out OOV alone does not measure tokenization quality: tokenizers that keep hashtags, mentions and "
                "URLs whole can show a higher OOV. Read it together with the units-kept-whole table.")


def domain_implications():
    st.header("Domain differences and downstream implications")
    social, biomedical = st.columns(2)
    with social:
        st.subheader("Social media")
        st.markdown(
            "Handles, hashtags, URLs, contractions, emoticons and repeated punctuation carry context. "
            "Whitespace can leave punctuation attached; generic rules can split a marker from its content. "
            "Tweet-aware rules preserve these spans. Sentiment140's noemoticon file contains few emoticons, "
            "so use the interactive examples to inspect those cases.")
    with biomedical:
        st.subheader("Biomedical text")
        st.markdown(
            "Drug names, hyphenated compounds, abbreviations, decimals and value-unit pairs need "
            "domain-aware boundaries. General-purpose rules can split these expressions; WordPiece divides "
            "rare terms into subwords by design. Biomedical lexicons and number-aware rules can preserve the "
            "full span while optionally retaining component tokens.")
    st.markdown(
        "**Downstream implications (qualitative):** Losing hashtags, emoticons or intensifying punctuation "
        "can remove sentiment evidence. Splitting a drug name or statistic can break biomedical entity "
        "matching and value interpretation. Preserve source spans and add normalized or component features "
        "when useful. Sentiment classification scores are measured below for Sentiment140; biomedical "
        "downstream performance was not evaluated, so those implications remain qualitative.")


# ============================================================= dashboard
def dashboard():
    st.title("Tokenization Studio")
    st.caption("A comparative workspace for social-media and biomedical text.")

    metrics = load(folder, "metrics_mean_std.csv", index_cols=[0, 1])
    if metrics is None:
        st.error(f"'{folder}/metrics_mean_std.csv' not found. Run tokenizer_experiments.py first.")
        return

    seeds = load(folder, "per_seed_metrics.csv", index_cols=None)
    n_seeds = seeds["seed"].nunique() if seeds is not None and "seed" in seeds else "?"

    all_dom = sorted(metrics["domain"].unique())
    all_tok = [t for t in ORDER if t in metrics["tokenizer"].values]
    domains = st.sidebar.multiselect("Dataset / Domain", all_dom, default=all_dom)
    toks = st.sidebar.multiselect("Tokenizer", all_tok, default=all_tok)
    m = metrics[metrics["domain"].isin(domains) & metrics["tokenizer"].isin(toks)]
    if m.empty:
        st.info("Select at least one dataset and tokenizer.")
        return

    units = load(folder, "units_kept_whole.csv", index_cols=None)
    abl = load(folder, "ablation_custom_tweet.csv", multi_header=True)
    ds = load(folder, "downstream_sentiment.csv", multi_header=True)
    pairwise = load(folder, "downstream_pairwise_tests.csv", index_cols=None)
    config_path = os.path.join(folder, "run_config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            run_config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        run_config = None

    overview_tab, token_tab, downstream_tab = st.tabs(
        ["Overview", "Token behavior", "Downstream"])

    with overview_tab:
        st.header("Study overview")
        c = st.columns(4)
        c[0].metric("Datasets", m["domain"].nunique())
        c[1].metric("Tokenizers", m["tokenizer"].nunique())
        c[2].metric("Seeds", n_seeds)
        c[3].metric("Tokens analyzed", f"{int(m['total_tokens_mean'].sum()):,}")

        with st.expander("Dataset details and methodology"):
            dataset_info()
        with st.expander("Reproducibility manifest"):
            if run_config is None:
                st.info("run_config.json is missing or invalid.")
            else:
                st.write(
                    f"**Samples per domain:** {run_config.get('samples_per_domain', 'unknown')}  ·  "
                    f"**Seeds:** {', '.join(map(str, run_config.get('seeds', [])))}  ·  "
                    f"**PubMed detokenized:** {run_config.get('pubmed_detokenized', 'unknown')}"
                )
                st.json(run_config, expanded=False)

        st.header("Tokenizer performance")
        table = pd.DataFrame({
            "Domain": m["domain"], "Tokenizer": m["tokenizer"],
            "Tokens/Doc": mean_std(m, "mean_tokens_doc"),
            "Held-out OOV %": mean_std(m, "oov_pct"),
            "Avg token length": mean_std(m, "avg_token_len"),
        })
        if "bert_unk_pct_mean" in m.columns:
            table["BERT [UNK] %"] = [
                f"{mean:.2f} ± {std:.2f}" if tokenizer == "WordPiece" and pd.notna(mean)
                else "not applicable"
                for tokenizer, mean, std in zip(m["tokenizer"], m["bert_unk_pct_mean"], m["bert_unk_pct_std"])
            ]
        st.dataframe(table, hide_index=True)
        download(table, "Download performance table", "performance_table.csv", key="dl_perf")
        grouped_bar(m, "mean_tokens_doc", "Mean tokens per document", "tokens / doc", 1)
        grouped_bar(m, "oov_pct", "Held-out token OOV rate", "unseen tokens (%)", 2)
        st.caption("Held-out OOV uses a tokenizer-specific vocabulary built from half the sample; BERT [UNK] is reported separately.")
        grouped_bar(m, "avg_token_len", "Average token length", "characters", 2)
        key_findings(m, units, domains)

    with token_tab:
        st.header("Token behavior by domain")
        length_distribution(toks)
        domain_comparison(m)
        if units is not None:
            st.header("Special units kept whole")
            dom = st.selectbox("Dataset", sorted(units["domain"].unique()), key="unit_dom")
            u = units[(units["domain"] == dom) & units["tokenizer"].isin(toks)]
            pv = u.pivot(index="tokenizer", columns="unit", values="kept_pct").reindex(
                [t for t in ORDER if t in u["tokenizer"].values])
            counts = u.drop_duplicates("unit").set_index("unit")["found"]
            pv = pv[[column for column in pv.columns if counts[column] > 0]]
            pv.columns = [f"{column} (n={int(counts[column])})" for column in pv.columns]
            st.dataframe(styled(pv))
            heatmap = pv.to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(max(9, len(pv.columns) * 0.9), max(3.5, len(pv.index) * 0.55)))
            image = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
            ax.set_xticks(np.arange(len(pv.columns)), labels=pv.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pv.index)), labels=pv.index)
            ax.set_title("Special units kept whole (%)")
            fig.colorbar(image, ax=ax, label="percent")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Unit counts are shown in the column labels. Chemical names are detected with an approximate pattern.")
            download(u, "Download units table", "units_kept_whole.csv", key="dl_units")
        else:
            st.info("units_kept_whole.csv is not available in this results folder.")
        domain_implications()
        with st.expander("Study limitations"):
            st.markdown(
                "- Sentiment140's no-emoticon training file contains few emoticons; conclusions about them rely on targeted examples.\n"
                "- Chemical names are identified with approximate patterns, not a curated biomedical gold lexicon.\n"
                "- Tokenizer-specific held-out OOV rates are not directly comparable to BERT's fixed-vocabulary [UNK] rate.\n"
                "- Downstream comparisons use repeated seeds and paired tests; interpret them as evidence for this sample and pipeline, not universal rankings.")

    with downstream_tab:
        st.header("Downstream evaluation")
        if ds is not None:
            d = pd.DataFrame({
                "Tokenizer": ds["tokenizer"],
                "Accuracy": mean_std(ds, "accuracy", 4),
                "Macro-F1": mean_std(ds, "macro_f1", 4),
                "Features": ds["n_features_mean"].round(0).astype(int),
            })
            st.dataframe(d, hide_index=True)
            scores = ds.set_index("tokenizer")[["accuracy_mean", "macro_f1_mean"]].reindex(
                [t for t in ORDER if t in ds["tokenizer"].values])
            st.bar_chart(scores, y_label="score")
            st.caption("TF-IDF (1-2 grams) + logistic regression on Sentiment140, mean over seeds.")
            if pairwise is not None:
                st.subheader("Paired tokenizer comparisons")
                st.caption("Paired t-tests across seeds; 95% confidence intervals for mean differences; Holm correction across all tests.")
                test_view = pairwise.drop(columns=["index"], errors="ignore").sort_values("p_value_holm")
                st.dataframe(test_view, hide_index=True)
            else:
                st.info("Pairwise tests are unavailable; rerun the updated experiment script.")
        else:
            st.info("No downstream results. Run tokenizer_experiments.py without --skip_downstream.")

        if abl is not None:
            st.header("Custom tweet-tokenizer ablation")
            removed_features = {
                "abbrev": "Abbreviation handling",
                "contraction": "Contraction handling",
                "emoji": "Emoji recognition",
                "emoticon": "Emoticon handling",
                "hashtag": "Hashtag handling",
                "hyphen": "Hyphenated-term handling",
                "mention": "Mention handling",
                "number": "Number handling",
                "plusminus": "Plus/minus handling",
                "repeat_punct": "Repeated-punctuation handling",
                "url": "URL handling",
            }
            abl = abl.copy()
            abl["Feature removed"] = abl["variant"].map(
                lambda variant: "None (full tokenizer)"
                if variant == "full"
                else removed_features.get(variant.removeprefix("-"), variant)
            )
            a = pd.DataFrame({
                "Feature removed": abl["Feature removed"],
                "Tokens/Doc": mean_std(abl, "mean_tokens_doc"),
                "OOV %": mean_std(abl, "oov_pct"),
                "Units kept %": mean_std(abl, "kept_pct"),
            })
            st.dataframe(a, hide_index=True)
            variants = abl.reset_index(drop=True)
            variant_order = variants["Feature removed"].tolist()
            for metric, title in (
                ("mean_tokens_doc_mean", "Tokens per document"),
                ("oov_pct_mean", "OOV (%)"),
                ("kept_pct_mean", "Units kept whole (%)"),
            ):
                chart = alt.Chart(variants).mark_bar().encode(
                    x=alt.X(f"{metric}:Q", title=title),
                    y=alt.Y("Feature removed:N", sort=variant_order, title="Feature removed",
                            axis=alt.Axis(labelLimit=200, labelOverlap=False)),
                    tooltip=[alt.Tooltip("Feature removed:N"),
                             alt.Tooltip(f"{metric}:Q", title=title, format=".2f")],
                ).properties(height=max(300, 28 * len(variant_order)))
                st.altair_chart(chart, width="stretch")
            st.caption("Each ablation removes the named feature from the custom tokenizer; compare the resulting metrics with the full tokenizer.")

# ============================================================== own text
def own_text():
    st.title("🔍 Analyze Your Own Text")
    st.caption("Enter your own social-media or biomedical text and compare how all seven tokenizers process it.")

    st.header("1️⃣ Select text domain")
    domain_choice = st.radio("Choose the type of text:", list(EXAMPLES), horizontal=True)

    st.header("2️⃣ Enter text")
    if "txt" not in st.session_state:
        st.session_state.txt = EXAMPLES[domain_choice]
    if st.button("Load example text"):
        st.session_state.txt = EXAMPLES[domain_choice]
    text = st.text_area("Enter a sentence, paragraph, tweet, or abstract sentence:", key="txt", height=140)

    if st.button("🔍 Analyze Text", type="primary"):
        if not text.strip():
            st.warning("Please enter some text before clicking Analyze.")
        else:
            tokenizers = get_tokenizers()
            clean = re.sub(r"\s+", " ", text).strip()
            st.session_state["analysis"] = {
                "text": clean, "raw_len": len(text), "social": "Social" in domain_choice,
                "out": {n: [str(t) for t in tokenizers[n](clean)] for n in ORDER},
            }

    a = st.session_state.get("analysis")        # kept in session so it survives other widget clicks
    if not a:
        return
    text, outputs = a["text"], a["out"]

    st.markdown("---")
    st.header("📄 Input Text")
    st.info(text)

    st.header("📌 Input Statistics")
    c = st.columns(3)
    c[0].metric("Characters", len(text))
    c[1].metric("Whitespace words", len(text.split()))
    c[2].metric("Estimated sentences", estimate_sentences(text))
    if a.get("raw_len") != len(text):
        st.caption(f"Extra spaces or line breaks were collapsed ({a['raw_len']} -> {len(text)} characters); "
                   "this does not change the tokens.")

    st.header("📊 Tokenizer comparison")
    rows = []
    for n, toks in outputs.items():
        rows.append({"Tokenizer": n, "Tokens": len(toks), "Unique": len({t.lower() for t in toks}),
                     "UNK": toks.count("[UNK]"),
                     "Avg token length": round(sum(map(len, toks)) / max(len(toks), 1), 2)})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index("Tokenizer")["Tokens"], y_label="tokens")
    download(df, "Download tokenization statistics", "tokenization_stats.csv", key="dl_own_stats")

    st.header("🧩 Are special units kept whole?")
    unit_rows, any_found = [], False
    for n, toks in outputs.items():
        row = {"Tokenizer": n}
        for kind, (f, kept) in kept_whole([text], [toks]).items():
            if f:
                any_found = True
                row[f"{kind} (n={f})"] = 100.0 * kept / f
        unit_rows.append(row)
    if any_found:
        udf = pd.DataFrame(unit_rows).set_index("Tokenizer").dropna(axis=1, how="all")
        st.dataframe(styled(udf, "{:.0f}"))
        st.caption("With a single input every cell is 0 or 100. Use the dashboard for full-sample percentages.")
    else:
        st.info("No URLs, mentions, hashtags, emoticons, decimals, hyphenated terms, abbreviations, "
                "chemical names or contractions were found in this text.")

    st.header("🔤 Detailed tokenization output")
    tabs = st.tabs(ORDER)
    for tab, n in zip(tabs, ORDER):
        with tab:
            st.write(f"**{len(outputs[n])} tokens**")
            st.code(" | ".join(outputs[n]) or "(none)", language=None)
    long_df = pd.DataFrame([{"Tokenizer": n, "Position": i + 1, "Token": t, "Length": len(t)}
                            for n, toks in outputs.items() for i, t in enumerate(toks)])
    if not long_df.empty:
        download(long_df, "Download all token outputs", "all_tokenizer_outputs.csv", key="dl_own_all")

    st.header("💡 Interpretation")
    counts = df.set_index("Tokenizer")["Tokens"]
    st.markdown(f"- **{counts.idxmax()}** produced the most tokens (**{counts.max()}**) and "
                f"**{counts.idxmin()}** the fewest (**{counts.min()}**).")
    unk = int(df.loc[df["Tokenizer"] == "WordPiece", "UNK"].sum())
    if unk:
        st.markdown(f"- WordPiece produced **{unk} [UNK]** token(s): characters missing from the BERT "
                    "vocabulary, such as emoji.")
    if a["social"]:
        st.markdown("- For social-media text, check hashtags, mentions, URLs, emoticons, elongated words and "
                    "repeated punctuation. Tweet-aware tokenizers (TweetTokenizer, CustomTweetRegex) are "
                    "designed to keep these whole.")
    else:
        st.markdown("- For biomedical text, check hyphenated chemical names, decimals, ± signs, dotted "
                    "abbreviations and acronyms. Generic tokenizers often split these; WordPiece splits rare "
                    "terms into sub-word pieces instead of producing an unknown word.")


if mode.startswith("📊"):
    dashboard()
else:
    own_text()