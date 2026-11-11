"""
Streamlit front end for the tokenization study (v2 results).

Needs tokenizer_experiments.py in the same folder (rename Tokenizer.py if
that is what you saved it as) and the CSV files made by that script.

Run:
  streamlit run app_v2.py --server.fileWatcherType none
"""
import os

import pandas as pd
import streamlit as st

from tokenizer_experiments import UNIT_PATTERNS, build_tokenizers, kept_whole

st.set_page_config(page_title="Text & Tokenization Analysis", page_icon="🔤", layout="wide")

ORDER = ["whitespace", "regex", "NLTK_word", "TweetTokenizer", "spaCy", "WordPiece", "CustomTweetRegex"]
EXAMPLE = ("I absolutely LOVE this product!!! 😍 Can't wait :) Check https://example.com "
           "#AmazingProduct @brand but the delivery was sooo slow!!!")


# ------------------------------------------------------------- helpers
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
        return pd.read_csv(path, index_col=index_cols).reset_index()
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read {name}: {e}")
        return None


def mean_std(df, base, nd=2):
    m, s = df[f"{base}_mean"], df.get(f"{base}_std")
    if s is None:
        return m.round(nd).astype(str)
    return m.round(nd).astype(str) + " ± " + s.round(nd).fillna(0).astype(str)


def pivot_chart(df, value, title, ylabel):
    st.subheader(title)
    p = df.pivot(index="tokenizer", columns="domain", values=value).reindex(
        [t for t in ORDER if t in df["tokenizer"].values])
    st.bar_chart(p, y_label=ylabel)


def download(df, label, fname):
    st.download_button(f"⬇️ {label}", df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


# -------------------------------------------------------------- sidebar
st.sidebar.title("🔤 Text Analysis")
mode = st.sidebar.radio("Select mode", ["📊 Results Dashboard", "🔍 Analyze Your Own Text"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Project**  \nComparative Study of Tokenization Methods on Social Media and Biomedical Text")
folder = st.sidebar.text_input("Results folder", "my_results_v2")


# ------------------------------------------------------------ dashboard
def dashboard():
    st.title("🔤 Text & Tokenization Analysis")
    st.caption("Comparative analysis of seven tokenization methods on social media and biomedical text.")

    metrics = load(folder, "metrics_mean_std.csv", index_cols=[0, 1])
    if metrics is None:
        st.error(f"'{folder}/metrics_mean_std.csv' not found. Run tokenizer_experiments.py first.")
        return

    seeds = load(folder, "per_seed_metrics.csv", index_cols=None)
    n_seeds = seeds["seed"].nunique() if seeds is not None and "seed" in seeds else "?"

    domains = st.sidebar.multiselect("Dataset / Domain", sorted(metrics["domain"].unique()),
                                     default=sorted(metrics["domain"].unique()))
    toks = st.sidebar.multiselect("Tokenizer", [t for t in ORDER if t in metrics["tokenizer"].values],
                                  default=[t for t in ORDER if t in metrics["tokenizer"].values])
    m = metrics[metrics["domain"].isin(domains) & metrics["tokenizer"].isin(toks)]
    if m.empty:
        st.info("Select at least one dataset and tokenizer.")
        return

    st.header("📌 Study Overview")
    c = st.columns(4)
    c[0].metric("Datasets", m["domain"].nunique())
    c[1].metric("Tokenizers", m["tokenizer"].nunique())
    c[2].metric("Seeds", n_seeds)
    c[3].metric("Tokens analysed (per seed)", f"{int(m['total_tokens_mean'].sum()):,}")

    st.header("📊 Tokenizer Performance (mean ± std over seeds)")
    table = pd.DataFrame({
        "Domain": m["domain"], "Tokenizer": m["tokenizer"],
        "Tokens/Doc": mean_std(m, "mean_tokens_doc"),
        "OOV %": mean_std(m, "oov_pct"),
        "Avg token length": mean_std(m, "avg_token_len"),
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    download(table, "Download performance table", "performance_table.csv")

    pivot_chart(m, "mean_tokens_doc_mean", "📈 Mean tokens per document", "tokens / doc")
    pivot_chart(m, "oov_pct_mean", "📉 OOV rate", "OOV %")
    st.caption("Vocabulary built from the first half of each sample by the same tokenizer; measured on the second half.")
    pivot_chart(m, "avg_token_len_mean", "📏 Average token length", "characters")

    # ---- units kept whole
    units = load(folder, "units_kept_whole.csv", index_cols=None)
    if units is not None:
        st.header("🧩 Are special units kept whole?")
        dom = st.selectbox("Dataset", sorted(units["domain"].unique()), key="unit_dom")
        u = units[units["domain"] == dom]
        pv = u.pivot(index="tokenizer", columns="unit", values="kept_pct").reindex(
            [t for t in ORDER if t in u["tokenizer"].values])
        counts = u.drop_duplicates("unit").set_index("unit")["found"]
        pv.columns = [f"{c} (n={int(counts[c])})" for c in pv.columns]
        try:
            st.dataframe(pv.style.format("{:.1f}", na_rep="–").background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                         use_container_width=True)
        except Exception:  # matplotlib missing -> plain table
            st.dataframe(pv, use_container_width=True)
        st.caption("% of units that the tokenizer outputs as exactly one token (counts summed over seeds). "
                   "'–' means the unit does not occur in that dataset.")
        download(u, "Download units table", "units_kept_whole.csv")

    # ---- ablation
    abl = load(folder, "ablation_custom_tweet.csv", multi_header=True)
    if abl is not None:
        st.header("🔬 CustomTweetRegex ablation (Sentiment140)")
        a = pd.DataFrame({
            "Variant": abl["variant"],
            "Tokens/Doc": mean_std(abl, "mean_tokens_doc"),
            "OOV %": mean_std(abl, "oov_pct"),
            "Units kept %": mean_std(abl, "kept_pct"),
        })
        st.dataframe(a, hide_index=True, use_container_width=True)
        st.caption("Each row removes one rule from the full tokenizer. Lower 'Units kept %' = that rule matters.")

    # ---- downstream
    ds = load(folder, "downstream_sentiment.csv", multi_header=True)
    if ds is not None:
        st.header("🎯 Downstream sentiment classification")
        d = pd.DataFrame({
            "Tokenizer": ds["tokenizer"],
            "Accuracy": mean_std(ds, "accuracy", 4),
            "Macro-F1": mean_std(ds, "macro_f1", 4),
            "Features": ds["n_features_mean"].round(0).astype(int),
        })
        st.dataframe(d, hide_index=True, use_container_width=True)
        st.bar_chart(ds.set_index("tokenizer")["accuracy_mean"].reindex(
            [t for t in ORDER if t in ds["tokenizer"].values]), y_label="accuracy")

    # ---- key findings (generated from the numbers)
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
    st.markdown("- OOV alone does not measure tokenization quality: tokenizers that keep hashtags, mentions and "
                "URLs whole can show a higher OOV. Read it together with the units-kept-whole table.")

    # ---- manual inspection
    man = None
    for name in ("manual_inspection_30.csv",):
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            p = os.path.join("my_results", name)
        if os.path.exists(p):
            man = pd.read_csv(p)
    if man is not None:
        st.header("🔎 Manual inspection examples")
        sel = st.selectbox("Examples from", ["Both"] + sorted(man["domain"].unique()))
        view = man if sel == "Both" else man[man["domain"] == sel]
        st.dataframe(view, hide_index=True, use_container_width=True)
        download(view, "Download manual inspection", "manual_inspection.csv")


# ------------------------------------------------------------ own text
def own_text():
    st.title("🔍 Analyze Your Own Text")
    st.caption("Enter your own social-media or biomedical text and compare how all seven tokenizers process it.")

    if "txt" not in st.session_state:
        st.session_state.txt = EXAMPLE
    if st.button("Load example text"):
        st.session_state.txt = EXAMPLE
    text = st.text_area("Enter a sentence, tweet, or abstract sentence:", key="txt", height=140)

    if not st.button("🔍 Analyze Text", type="primary") or not text.strip():
        return

    tokenizers = get_tokenizers()
    outputs = {n: tokenizers[n](text) for n in ORDER}

    c = st.columns(3)
    c[0].metric("Characters", len(text))
    c[1].metric("Whitespace words", len(text.split()))
    c[2].metric("Estimated sentences", max(1, sum(text.count(x) for x in ".!?") and len(
        [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])))

    st.header("📊 Tokenizer comparison")
    rows = []
    for n, toks in outputs.items():
        rows.append({"Tokenizer": n, "Tokens": len(toks), "Unique": len(set(t.lower() for t in toks)),
                     "Avg token length": round(sum(map(len, toks)) / max(len(toks), 1), 2)})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("Tokenizer")["Tokens"], y_label="tokens")
    download(df, "Download tokenization statistics", "tokenization_stats.csv")

    st.header("🧩 Are special units kept whole?")
    found_any = False
    unit_rows = []
    for n, toks in outputs.items():
        k = kept_whole([text], [toks])
        row = {"Tokenizer": n}
        for kind, (f, kept) in k.items():
            if f:
                found_any = True
                row[f"{kind} (n={f})"] = 100.0 * kept / f
        unit_rows.append(row)
    if found_any:
        udf = pd.DataFrame(unit_rows).set_index("Tokenizer")
        udf = udf.dropna(axis=1, how="all")
        try:
            st.dataframe(udf.style.format("{:.0f}", na_rep="–").background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                         use_container_width=True)
        except Exception:  # noqa: BLE001
            st.dataframe(udf, use_container_width=True)
        st.caption("With a single input every cell is 0 or 100. Use the dashboard for full-sample percentages.")
    else:
        st.info("No URLs, mentions, hashtags, emoticons, decimals, hyphenated terms, abbreviations or "
                "contractions were found in this text.")

    st.header("🔤 Detailed tokenization output")
    tabs = st.tabs(ORDER)
    for tab, n in zip(tabs, ORDER):
        with tab:
            st.write(f"**{len(outputs[n])} tokens**")
            st.code(" | ".join(outputs[n]), language=None)


if mode.startswith("📊"):
    dashboard()
else:
    own_text()