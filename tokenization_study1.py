import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import html
import csv
import io

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Text & Tokenization Analysis",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "my_results"
SUMMARY_FILE = f"{RESULTS_DIR}/summary_metrics.csv"
MANUAL_FILE = f"{RESULTS_DIR}/manual_inspection_examples.csv"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .token-box {
        padding: 8px 10px;
        margin: 4px;
        border-radius: 8px;
        border: 1px solid #d9d9d9;
        display: inline-block;
        background: #f7f7f7;
        font-family: monospace;
        font-size: 0.9rem;
    }

    .result-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #fafafa;
    }

    .small-note {
        color: #666;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e0e0e0;
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_summary():
    try:
        return pd.read_csv(SUMMARY_FILE)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_manual_examples():
    try:
        return pd.read_csv(MANUAL_FILE)
    except Exception:
        return pd.DataFrame()


def safe_regex_tokenize(text):
    """
    Fallback regex tokenizer.
    Keeps words, contractions, URLs, mentions, hashtags,
    numbers and repeated punctuation reasonably intact.
    """
    pattern = (
        r"https?://\S+"
        r"|www\.\S+"
        r"|[@#][A-Za-z0-9_]+"
        r"|[A-Za-z]+(?:'[A-Za-z]+)?"
        r"|\d+(?:\.\d+)?"
        r"|[!?.,;:]+"
        r"|[^\s]"
    )

    return re.findall(pattern, text)


def whitespace_tokenize(text):
    return text.split()


def nltk_word_tokenize(text):
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except Exception:
        return safe_regex_tokenize(text)


def tweet_tokenize(text):
    try:
        from nltk.tokenize import TweetTokenizer
        tokenizer = TweetTokenizer(
            preserve_case=True,
            reduce_len=False,
            strip_handles=False
        )
        return tokenizer.tokenize(text)
    except Exception:
        return safe_regex_tokenize(text)


@st.cache_resource
def load_spacy_model():
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return spacy.blank("en")
    except Exception:
        return None


def spacy_tokenize(text):
    nlp = load_spacy_model()

    if nlp is None:
        return safe_regex_tokenize(text)

    doc = nlp(text)
    return [token.text for token in doc]


@st.cache_resource
def load_wordpiece_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True
        )
    except Exception:
        return None


def wordpiece_tokenize(text):
    tokenizer = load_wordpiece_tokenizer()

    if tokenizer is None:
        return safe_regex_tokenize(text)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    return tokenizer.convert_ids_to_tokens(encoded["input_ids"])


def custom_tweet_regex_tokenize(text):
    """
    Custom tokenizer designed for social-media style text.
    """
    pattern = (
        r"https?://[^\s]+"
        r"|www\.[^\s]+"
        r"|@[A-Za-z0-9_]+"
        r"|#[A-Za-z0-9_]+"
        r"|[A-Za-z]+(?:['’][A-Za-z]+)*"
        r"|\d+(?:[.,]\d+)*"
        r"|[!?]+"
        r"|[.,;:]+"
        r"|[-–—]+"
        r"|[^\w\s]"
    )

    return re.findall(pattern, text)


def get_manual_tokenizers():
    """
    Returns the seven tokenizers used by the study.
    """
    return {
        "Whitespace": whitespace_tokenize,
        "Regex": safe_regex_tokenize,
        "NLTK_word": nltk_word_tokenize,
        "TweetTokenizer": tweet_tokenize,
        "spaCy": spacy_tokenize,
        "WordPiece": wordpiece_tokenize,
        "CustomTweetRegex": custom_tweet_regex_tokenize,
    }


def calculate_text_statistics(tokens):
    if not tokens:
        return {
            "Token Count": 0,
            "Unique Tokens": 0,
            "Average Token Length": 0,
            "Minimum Token Length": 0,
            "Maximum Token Length": 0,
            "Median Token Length": 0
        }

    lengths = [len(str(token)) for token in tokens]

    return {
        "Token Count": len(tokens),
        "Unique Tokens": len(set(tokens)),
        "Average Token Length": round(float(np.mean(lengths)), 2),
        "Minimum Token Length": min(lengths),
        "Maximum Token Length": max(lengths),
        "Median Token Length": round(float(np.median(lengths)), 2)
    }


def tokens_to_html(tokens):
    if not tokens:
        return "<span class='small-note'>No tokens generated.</span>"

    pieces = []

    for token in tokens:
        safe = html.escape(str(token))
        pieces.append(
            f"<span class='token-box'>{safe}</span>"
        )

    return "".join(pieces)


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔤 Text Analysis")

page = st.sidebar.radio(
    "Select Mode",
    [
        "📊 Results Dashboard",
        "🔍 Analyze Your Own Text"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Project**

    Comparative Study of Tokenization Methods on
    Social Media and Biomedical Text

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

# ============================================================
# PAGE 1 — RESULTS DASHBOARD
# ============================================================

if page == "📊 Results Dashboard":

    st.markdown(
        "<div class='main-title'>🔤 Text & Tokenization Analysis</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Comparative analysis of seven tokenization methods "
        "on social media and biomedical text.</div>",
        unsafe_allow_html=True
    )

    summary = load_summary()
    manual_examples = load_manual_examples()

    if summary.empty:
        st.error(
            "summary_metrics.csv was not found. "
            "Please run tokenization_study.py first."
        )
        st.stop()

    # --------------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader("Dashboard Filters")

    available_domains = list(summary["domain"].dropna().unique())

    selected_domains = st.sidebar.multiselect(
        "Dataset / Domain",
        available_domains,
        default=available_domains
    )

    available_tokenizers = list(
        summary["tokenizer"].dropna().unique()
    )

    selected_tokenizers = st.sidebar.multiselect(
        "Tokenizer",
        available_tokenizers,
        default=available_tokenizers
    )

    filtered = summary[
        summary["domain"].isin(selected_domains)
        & summary["tokenizer"].isin(selected_tokenizers)
    ].copy()

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>📌 Study Overview</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Datasets",
            summary["domain"].nunique()
        )

    with c2:
        st.metric(
            "Tokenizers",
            summary["tokenizer"].nunique()
        )

    with c3:
        st.metric(
            "Samples / Dataset",
            f"{int(summary['documents'].max()):,}"
        )

    with c4:
        st.metric(
            "Total Analysis Rows",
            f"{len(summary):,}"
        )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>📚 Dataset Information</div>",
        unsafe_allow_html=True
    )

    dataset_col1, dataset_col2 = st.columns(2)

    with dataset_col1:
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
            - Emoticons
            """
        )

    with dataset_col2:
        st.markdown(
            """
            ### 🧬 PubMed RCT

            Biomedical research sentences extracted from
            randomized controlled trial abstracts.

            **Characteristics**
            - Scientific terminology
            - Long sentences
            - Medical terminology
            - Numbers
            - Abbreviations
            - Technical vocabulary
            """
        )

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>📊 Tokenizer Performance</div>",
        unsafe_allow_html=True
    )

    if filtered.empty:
        st.warning("No data matches the selected filters.")
    else:
        display_columns = [
            "domain",
            "tokenizer",
            "documents",
            "mean_tokens_per_document",
            "median_tokens_per_document",
            "mean_token_length_chars",
            "token_length_p50_chars",
            "token_length_p90_chars",
            "oov_rate_tokens",
            "oov_rate_types",
            "unk_rate"
        ]

        table = filtered[display_columns].copy()

        table["oov_rate_tokens"] = (
            table["oov_rate_tokens"] * 100
        ).round(2)

        table["oov_rate_types"] = (
            table["oov_rate_types"] * 100
        ).round(2)

        table["unk_rate"] = (
            table["unk_rate"] * 100
        ).round(4)

        table = table.rename(
            columns={
                "domain": "Domain",
                "tokenizer": "Tokenizer",
                "documents": "Documents",
                "mean_tokens_per_document": "Mean Tokens/Doc",
                "median_tokens_per_document": "Median Tokens/Doc",
                "mean_token_length_chars": "Mean Token Length",
                "token_length_p50_chars": "P50 Length",
                "token_length_p90_chars": "P90 Length",
                "oov_rate_tokens": "OOV Tokens (%)",
                "oov_rate_types": "OOV Types (%)",
                "unk_rate": "UNK Rate (%)"
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Performance Table",
            data=dataframe_to_csv_bytes(table),
            file_name="tokenizer_performance.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------
    # CHART 1
    # --------------------------------------------------------

    if not filtered.empty:

        st.markdown(
            "<div class='section-title'>📈 Mean Tokens per Document</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        chart_data = filtered.copy()

        chart_data["label"] = (
            chart_data["domain"]
            + " — "
            + chart_data["tokenizer"]
        )

        ax.bar(
            chart_data["label"],
            chart_data["mean_tokens_per_document"]
        )

        ax.set_ylabel("Mean Tokens per Document")
        ax.set_xlabel("Dataset — Tokenizer")
        ax.set_title("Average Number of Tokens Generated")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # CHART 2
    # --------------------------------------------------------

    if not filtered.empty:

        st.markdown(
            "<div class='section-title'>📉 OOV Rate</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        chart_data = filtered.copy()

        chart_data["label"] = (
            chart_data["domain"]
            + " — "
            + chart_data["tokenizer"]
        )

        ax.bar(
            chart_data["label"],
            chart_data["oov_rate_tokens"] * 100
        )

        ax.set_ylabel("OOV Rate (%)")
        ax.set_xlabel("Dataset — Tokenizer")
        ax.set_title("Out-of-Vocabulary Token Rate")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # CHART 3
    # --------------------------------------------------------

    if not filtered.empty:

        st.markdown(
            "<div class='section-title'>📏 Average Token Length</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        chart_data = filtered.copy()

        chart_data["label"] = (
            chart_data["domain"]
            + " — "
            + chart_data["tokenizer"]
        )

        ax.bar(
            chart_data["label"],
            chart_data["mean_token_length_chars"]
        )

        ax.set_ylabel("Characters")
        ax.set_xlabel("Dataset — Tokenizer")
        ax.set_title("Average Token Length")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # DOMAIN COMPARISON
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>🔬 Domain Comparison</div>",
        unsafe_allow_html=True
    )

    domain_summary = (
        summary
        .groupby("domain")
        .agg(
            Mean_Tokens_Per_Document=(
                "mean_tokens_per_document",
                "mean"
            ),
            Mean_Token_Length=(
                "mean_token_length_chars",
                "mean"
            ),
            OOV_Rate=(
                "oov_rate_tokens",
                "mean"
            )
        )
        .reset_index()
    )

    domain_summary["OOV_Rate"] = (
        domain_summary["OOV_Rate"] * 100
    ).round(2)

    domain_summary["Mean_Tokens_Per_Document"] = (
        domain_summary["Mean_Tokens_Per_Document"]
        .round(2)
    )

    domain_summary["Mean_Token_Length"] = (
        domain_summary["Mean_Token_Length"]
        .round(2)
    )

    domain_summary = domain_summary.rename(
        columns={
            "domain": "Domain",
            "Mean_Tokens_Per_Document": "Average Tokens/Doc",
            "Mean_Token_Length": "Average Token Length",
            "OOV_Rate": "Average OOV (%)"
        }
    )

    st.dataframe(
        domain_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>💡 Key Findings</div>",
        unsafe_allow_html=True
    )

    wp = summary[
        summary["tokenizer"].str.lower() == "wordpiece"
    ]

    whitespace = summary[
        summary["tokenizer"].str.lower() == "whitespace"
    ]

    if not wp.empty:

        wp_sent = wp[
            wp["domain"] == "Sentiment140"
        ]

        wp_pub = wp[
            wp["domain"] == "PubMed RCT"
        ]

        if not wp_sent.empty:
            st.write(
                f"• **WordPiece** achieves approximately "
                f"**{wp_sent.iloc[0]['oov_rate_tokens'] * 100:.2f}%** "
                f"OOV token rate on Sentiment140."
            )

        if not wp_pub.empty:
            st.write(
                f"• **WordPiece** achieves approximately "
                f"**{wp_pub.iloc[0]['oov_rate_tokens'] * 100:.2f}%** "
                f"OOV token rate on PubMed RCT."
            )

    if not whitespace.empty:

        ws_sent = whitespace[
            whitespace["domain"] == "Sentiment140"
        ]

        if not ws_sent.empty:
            st.write(
                f"• Whitespace tokenization has a relatively high "
                f"OOV rate of approximately "
                f"**{ws_sent.iloc[0]['oov_rate_tokens'] * 100:.2f}%** "
                f"on Sentiment140."
            )

    st.write(
        "• WordPiece generally produces more tokens because rare or "
        "unseen words can be split into smaller sub-word units."
    )

    st.write(
        "• Tweet-oriented tokenizers are useful for social-media text "
        "because they better handle hashtags, mentions, URLs and "
        "informal expressions."
    )

    st.write(
        "• PubMed RCT generally produces more tokens per document "
        "because biomedical sentences are longer and contain technical "
        "terminology."
    )

    # --------------------------------------------------------
    # MANUAL INSPECTION
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>🔎 Manual Inspection Examples</div>",
        unsafe_allow_html=True
    )

    if manual_examples.empty:
        st.info(
            "manual_inspection_examples.csv was not found."
        )
    else:

        st.dataframe(
            manual_examples,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Manual Inspection",
            data=dataframe_to_csv_bytes(manual_examples),
            file_name="manual_inspection_examples.csv",
            mime="text/csv"
        )


# ============================================================
# PAGE 2 — ANALYZE USER TEXT
# ============================================================

else:

    st.markdown(
        "<div class='main-title'>🔍 Analyze Your Own Text</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Enter your own social-media or biomedical "
        "text and compare how all seven tokenizers process it.</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DOMAIN SELECTION
    # --------------------------------------------------------

    st.markdown(
        "<div class='section-title'>1️⃣ Select Text Domain</div>",
        unsafe_allow_html=True
    )

    domain = st.radio(
        "Choose the type of text:",
        [
            "🐦 Social Media / Sentiment140",
            "🧬 Biomedical / PubMed RCT"
        ],
        horizontal=True
    )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    if "Social Media" in domain:

        example_text = (
            "I absolutely LOVE this product!!! 😍 "
            "Check https://example.com #AmazingProduct @brand "
            "but the delivery was sooo slow!!!"
        )

    else:

        example_text = (
            "The intervention significantly improved clinical outcomes "
            "among patients with type 2 diabetes mellitus."
        )

    st.markdown(
        "<div class='section-title'>2️⃣ Enter Text</div>",
        unsafe_allow_html=True
    )

    use_example = st.checkbox(
        "Use example text",
        value=False
    )

    if use_example:
        default_text = example_text
    else:
        default_text = ""

    user_text = st.text_area(
        "Enter a sentence, paragraph, tweet, abstract sentence, or other text:",
        value=default_text,
        height=160,
        placeholder="Type or paste your text here..."
    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analyze Text",
        type="primary",
        use_container_width=True
    )

    if analyze:

        if not user_text.strip():

            st.warning(
                "Please enter some text before clicking Analyze."
            )
            st.stop()

        st.markdown("---")

        st.markdown(
            "<div class='section-title'>📄 Input Text</div>",
            unsafe_allow_html=True
        )

        st.info(user_text)

        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        tokenizers = get_manual_tokenizers()

        results = {}
        stats_rows = []

        progress = st.progress(0)

        tokenizer_items = list(tokenizers.items())

        for i, (name, tokenizer_function) in enumerate(
            tokenizer_items
        ):

            try:

                tokens = tokenizer_function(user_text)

                if tokens is None:
                    tokens = []

                tokens = [str(x) for x in tokens]

            except Exception as e:

                tokens = []

                st.warning(
                    f"{name} could not process the text: {e}"
                )

            results[name] = tokens

            stats = calculate_text_statistics(tokens)

            stats_rows.append(
                {
                    "Tokenizer": name,
                    "Token Count": stats["Token Count"],
                    "Unique Tokens": stats["Unique Tokens"],
                    "Average Token Length": stats[
                        "Average Token Length"
                    ],
                    "Median Token Length": stats[
                        "Median Token Length"
                    ],
                    "Minimum Length": stats[
                        "Minimum Token Length"
                    ],
                    "Maximum Length": stats[
                        "Maximum Token Length"
                    ]
                }
            )

            progress.progress(
                int((i + 1) / len(tokenizer_items) * 100)
            )

        progress.empty()

        stats_df = pd.DataFrame(stats_rows)

        # ----------------------------------------------------
        # BASIC TEXT STATISTICS
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>📌 Input Statistics</div>",
            unsafe_allow_html=True
        )

        word_count = len(user_text.split())
        character_count = len(user_text)
        sentence_count = max(
            1,
            len(
                re.findall(
                    r"[.!?]+",
                    user_text
                )
            )
        )

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Characters",
                character_count
            )

        with b:
            st.metric(
                "Whitespace Words",
                word_count
            )

        with c:
            st.metric(
                "Estimated Sentences",
                sentence_count
            )

        # ----------------------------------------------------
        # COMPARISON TABLE
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>📊 Tokenizer Comparison</div>",
            unsafe_allow_html=True
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD RESULTS
        # ----------------------------------------------------

        download_df = stats_df.copy()

        st.download_button(
            "⬇️ Download Tokenization Statistics",
            data=dataframe_to_csv_bytes(download_df),
            file_name="manual_tokenization_results.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # TOKEN COUNT CHART
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>📈 Token Count Comparison</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.bar(
            stats_df["Tokenizer"],
            stats_df["Token Count"]
        )

        ax.set_xlabel("Tokenizer")
        ax.set_ylabel("Number of Tokens")
        ax.set_title("Tokens Generated by Each Tokenizer")

        plt.xticks(
            rotation=40,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # TOKEN LENGTH CHART
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>📏 Average Token Length</div>",
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.bar(
            stats_df["Tokenizer"],
            stats_df["Average Token Length"]
        )

        ax.set_xlabel("Tokenizer")
        ax.set_ylabel("Characters")
        ax.set_title("Average Token Length")

        plt.xticks(
            rotation=40,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # ----------------------------------------------------
        # DETAILED TOKEN OUTPUT
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>🔤 Detailed Tokenization Output</div>",
            unsafe_allow_html=True
        )

        st.write(
            "Select a tokenizer to inspect the exact tokens generated."
        )

        selected_tokenizer = st.selectbox(
            "Tokenizer",
            list(results.keys())
        )

        selected_tokens = results[selected_tokenizer]

        st.markdown(
            f"### {selected_tokenizer}"
        )

        st.write(
            f"**{len(selected_tokens)} tokens generated**"
        )

        # Token chips
        st.markdown(
            tokens_to_html(selected_tokens),
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # RAW TOKEN OUTPUT
        # ----------------------------------------------------

        st.markdown(
            "### Raw Token Output"
        )

        st.code(
            str(selected_tokens),
            language="text"
        )

        # ----------------------------------------------------
        # TOKEN TABLE
        # ----------------------------------------------------

        if selected_tokens:

            token_table = pd.DataFrame(
                {
                    "Position": range(
                        1,
                        len(selected_tokens) + 1
                    ),
                    "Token": selected_tokens,
                    "Length": [
                        len(token)
                        for token in selected_tokens
                    ]
                }
            )

            st.dataframe(
                token_table,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Selected Tokenizer Output",
                data=dataframe_to_csv_bytes(token_table),
                file_name=(
                    selected_tokenizer
                    .replace(" ", "_")
                    + "_tokens.csv"
                ),
                mime="text/csv"
            )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>💡 Interpretation</div>",
            unsafe_allow_html=True
        )

        token_counts = stats_df.set_index(
            "Tokenizer"
        )["Token Count"]

        max_tokenizer = token_counts.idxmax()
        min_tokenizer = token_counts.idxmin()

        st.write(
            f"• **{max_tokenizer}** produced the highest number "
            f"of tokens: **{token_counts[max_tokenizer]}**."
        )

        st.write(
            f"• **{min_tokenizer}** produced the lowest number "
            f"of tokens: **{token_counts[min_tokenizer]}**."
        )

        if selected_tokenizer == "WordPiece":

            st.write(
                "• WordPiece uses sub-word tokenization. "
                "Therefore, uncommon or unknown words may be split "
                "into smaller pieces instead of being treated as "
                "one complete unknown word."
            )

        elif selected_tokenizer in [
            "TweetTokenizer",
            "CustomTweetRegex"
        ]:

            st.write(
                "• This tokenizer is particularly useful for "
                "social-media text because it can preserve or "
                "recognize structures such as hashtags, mentions, "
                "URLs and informal expressions."
            )

        elif selected_tokenizer == "Whitespace":

            st.write(
                "• Whitespace tokenization is simple and fast, "
                "but punctuation, URLs and attached symbols may "
                "remain part of the same token."
            )

        elif selected_tokenizer == "spaCy":

            st.write(
                "• spaCy performs linguistically informed "
                "tokenization and generally separates punctuation "
                "and textual components more systematically."
            )

        elif selected_tokenizer == "NLTK_word":

            st.write(
                "• NLTK word tokenization provides a more structured "
                "word-level representation than simple whitespace "
                "splitting."
            )

        else:

            st.write(
                "• Regex tokenization explicitly applies "
                "pattern-based rules to identify words, numbers, "
                "URLs, punctuation and other text structures."
            )

        # ----------------------------------------------------
        # DOMAIN-SPECIFIC INTERPRETATION
        # ----------------------------------------------------

        if "Social Media" in domain:

            st.markdown("### 🐦 Social Media Interpretation")

            st.write(
                "For social-media text, pay particular attention to "
                "hashtags, mentions, URLs, repeated punctuation, "
                "emojis and informal spellings."
            )

            st.write(
                "TweetTokenizer and CustomTweetRegex are designed "
                "to handle many of these structures more naturally."
            )

        else:

            st.markdown("### 🧬 Biomedical Interpretation")

            st.write(
                "For biomedical text, rare technical terms and "
                "specialized vocabulary are important."
            )

            st.write(
                "WordPiece can be useful because uncommon biomedical "
                "words can be divided into smaller sub-word units."
            )

        # ----------------------------------------------------
        # ALL TOKENIZER OUTPUTS
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>📋 All 7 Tokenizer Outputs</div>",
            unsafe_allow_html=True
        )

        for tokenizer_name, tokens in results.items():

            with st.expander(
                f"{tokenizer_name} — {len(tokens)} tokens"
            ):

                st.markdown(
                    tokens_to_html(tokens),
                    unsafe_allow_html=True
                )

                st.code(
                    str(tokens),
                    language="text"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Comparative Study of Tokenization Methods on Social Media "
    "and Biomedical Text | M.Tech Project"
)
