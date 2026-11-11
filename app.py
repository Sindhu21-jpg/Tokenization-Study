import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tokenization Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "my_results"

SUMMARY_FILE = RESULTS_DIR / "summary_metrics.csv"
EXAMPLES_FILE = RESULTS_DIR / "manual_inspection_examples.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        text-align: center;
        min-height: 120px;
    }

    .metric-title {
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_summary():
    if not SUMMARY_FILE.exists():
        return None

    return pd.read_csv(SUMMARY_FILE)


@st.cache_data
def load_examples():
    if not EXAMPLES_FILE.exists():
        return None

    return pd.read_csv(EXAMPLES_FILE)


summary_df = load_summary()
examples_df = load_examples()


# ============================================================
# CHECK FILES
# ============================================================

if summary_df is None:

    st.error(
        "summary_metrics.csv was not found."
    )

    st.info(
        f"Expected location:\n\n{SUMMARY_FILE}"
    )

    st.stop()


if examples_df is None:

    st.warning(
        "manual_inspection_examples.csv was not found."
    )

    st.info(
        f"Expected location:\n\n{EXAMPLES_FILE}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

summary_df.columns = summary_df.columns.str.strip()

if examples_df is not None:
    examples_df.columns = examples_df.columns.str.strip()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">🧠 Tokenization Analysis Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Comparative Study of Tokenization Methods on Social Media and Biomedical Text'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown("### Dataset")

available_domains = summary_df["domain"].dropna().unique().tolist()

selected_domain = st.sidebar.selectbox(
    "Select Domain",
    ["All Domains"] + available_domains
)


st.sidebar.markdown("### Tokenizer")

available_tokenizers = (
    summary_df["tokenizer"]
    .dropna()
    .unique()
    .tolist()
)

selected_tokenizers = st.sidebar.multiselect(
    "Select Tokenizers",
    available_tokenizers,
    default=available_tokenizers
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = summary_df.copy()

if selected_domain != "All Domains":
    filtered_df = filtered_df[
        filtered_df["domain"] == selected_domain
    ]

if selected_tokenizers:
    filtered_df = filtered_df[
        filtered_df["tokenizer"].isin(selected_tokenizers)
    ]


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Overview</div>',
    unsafe_allow_html=True
)

total_documents = int(summary_df["documents"].sum())

number_domains = summary_df["domain"].nunique()

number_tokenizers = summary_df["tokenizer"].nunique()

best_row = summary_df.loc[
    summary_df["oov_rate_tokens"].idxmin()
]

best_tokenizer = best_row["tokenizer"]
best_domain = best_row["domain"]

best_oov = best_row["oov_rate_tokens"] * 100

highest_token_row = summary_df.loc[
    summary_df["mean_tokens_per_document"].idxmax()
]

highest_tokenizer = highest_token_row["tokenizer"]

highest_tokens = highest_token_row["mean_tokens_per_document"]


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Documents</div>
            <div class="metric-value">{total_documents:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Domains</div>
            <div class="metric-value">{number_domains}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Tokenizers</div>
            <div class="metric-value">{number_tokenizers}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Best OOV Rate</div>
            <div class="metric-value">{best_oov:.2f}%</div>
            <div>{best_tokenizer}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📚 Dataset Information</div>',
    unsafe_allow_html=True
)

info1, info2 = st.columns(2)

with info1:

    st.markdown(
        """
        <div class="info-box">
        <b>Sentiment140</b><br>
        Social-media text containing hashtags, mentions,
        URLs, emoticons, emojis, informal language and
        repeated punctuation.
        </div>
        """,
        unsafe_allow_html=True
    )

with info2:

    st.markdown(
        """
        <div class="info-box">
        <b>PubMed RCT</b><br>
        Biomedical research text containing technical,
        scientific and domain-specific terminology.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

st.markdown(
    '<div class="section-title">📋 Tokenizer Performance</div>',
    unsafe_allow_html=True
)

display_df = filtered_df.copy()

display_df["oov_rate_tokens"] = (
    display_df["oov_rate_tokens"] * 100
)

display_df["oov_rate_types"] = (
    display_df["oov_rate_types"] * 100
)

display_df["unk_rate"] = (
    display_df["unk_rate"] * 100
)

display_df = display_df.rename(
    columns={
        "domain": "Domain",
        "tokenizer": "Tokenizer",
        "documents": "Documents",
        "mean_tokens_per_document": "Avg Tokens/Doc",
        "median_tokens_per_document": "Median Tokens/Doc",
        "mean_token_length_chars": "Avg Token Length",
        "token_length_p50_chars": "Token Length P50",
        "token_length_p90_chars": "Token Length P90",
        "oov_rate_tokens": "OOV Rate (%)",
        "oov_rate_types": "OOV Types (%)",
        "unk_rate": "UNK Rate (%)",
        "total_tokens": "Total Tokens"
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Avg Tokens/Doc": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Median Tokens/Doc": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Avg Token Length": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Token Length P50": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Token Length P90": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "OOV Rate (%)": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "OOV Types (%)": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
        "UNK Rate (%)": st.column_config.NumberColumn(
            format="%.4f%%"
        )
    }
)


# ============================================================
# CHART 1 — AVERAGE TOKENS PER DOCUMENT
# ============================================================

st.markdown(
    '<div class="section-title">📈 Average Tokens per Document</div>',
    unsafe_allow_html=True
)

fig1, ax1 = plt.subplots(figsize=(11, 5))

if not filtered_df.empty:

    chart_df = filtered_df.copy()

    for domain in chart_df["domain"].unique():

        domain_data = chart_df[
            chart_df["domain"] == domain
        ]

        ax1.plot(
            domain_data["tokenizer"],
            domain_data["mean_tokens_per_document"],
            marker="o",
            label=domain
        )

    ax1.set_xlabel("Tokenizer")
    ax1.set_ylabel("Average Tokens per Document")
    ax1.set_title("Average Tokens per Document by Tokenizer")
    ax1.tick_params(axis="x", rotation=35)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig1)

    plt.close(fig1)


# ============================================================
# CHART 2 — OOV RATE
# ============================================================

st.markdown(
    '<div class="section-title">📉 OOV Rate Comparison</div>',
    unsafe_allow_html=True
)

fig2, ax2 = plt.subplots(figsize=(11, 5))

if not filtered_df.empty:

    chart_df = filtered_df.copy()

    for domain in chart_df["domain"].unique():

        domain_data = chart_df[
            chart_df["domain"] == domain
        ]

        ax2.plot(
            domain_data["tokenizer"],
            domain_data["oov_rate_tokens"] * 100,
            marker="o",
            label=domain
        )

    ax2.set_xlabel("Tokenizer")
    ax2.set_ylabel("OOV Rate (%)")
    ax2.set_title("Out-of-Vocabulary Rate by Tokenizer")
    ax2.tick_params(axis="x", rotation=35)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


# ============================================================
# CHART 3 — TOKEN LENGTH
# ============================================================

st.markdown(
    '<div class="section-title">📏 Average Token Length</div>',
    unsafe_allow_html=True
)

fig3, ax3 = plt.subplots(figsize=(11, 5))

if not filtered_df.empty:

    chart_df = filtered_df.copy()

    for domain in chart_df["domain"].unique():

        domain_data = chart_df[
            chart_df["domain"] == domain
        ]

        ax3.plot(
            domain_data["tokenizer"],
            domain_data["mean_token_length_chars"],
            marker="o",
            label=domain
        )

    ax3.set_xlabel("Tokenizer")
    ax3.set_ylabel("Average Token Length (characters)")
    ax3.set_title("Average Token Length")
    ax3.tick_params(axis="x", rotation=35)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig3)

    plt.close(fig3)


# ============================================================
# CHART 4 — TOKEN LENGTH P50/P90
# ============================================================

st.markdown(
    '<div class="section-title">📊 Token Length Distribution</div>',
    unsafe_allow_html=True
)

distribution_df = filtered_df[
    [
        "domain",
        "tokenizer",
        "token_length_p50_chars",
        "token_length_p90_chars"
    ]
].copy()

distribution_df = distribution_df.rename(
    columns={
        "domain": "Domain",
        "tokenizer": "Tokenizer",
        "token_length_p50_chars": "P50",
        "token_length_p90_chars": "P90"
    }
)

st.dataframe(
    distribution_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOMAIN COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Domain Comparison</div>',
    unsafe_allow_html=True
)

if number_domains >= 2:

    domain_pivot = summary_df.pivot(
        index="tokenizer",
        columns="domain",
        values="oov_rate_tokens"
    ) * 100

    domain_pivot = domain_pivot.round(2)

    st.markdown("**OOV Rate (%) by Domain**")

    st.dataframe(
        domain_pivot,
        use_container_width=True
    )


# ============================================================
# BEST TOKENIZER
# ============================================================

st.markdown(
    '<div class="section-title">🏆 Key Findings</div>',
    unsafe_allow_html=True
)

sentiment_df = summary_df[
    summary_df["domain"] == "Sentiment140"
]

pubmed_df = summary_df[
    summary_df["domain"] == "PubMed RCT"
]


if not sentiment_df.empty:

    sentiment_best = sentiment_df.loc[
        sentiment_df["oov_rate_tokens"].idxmin()
    ]

    st.success(
        f"Sentiment140: {sentiment_best['tokenizer']} "
        f"achieved the lowest OOV rate "
        f"({sentiment_best['oov_rate_tokens'] * 100:.2f}%)."
    )


if not pubmed_df.empty:

    pubmed_best = pubmed_df.loc[
        pubmed_df["oov_rate_tokens"].idxmin()
    ]

    st.success(
        f"PubMed RCT: {pubmed_best['tokenizer']} "
        f"achieved the lowest OOV rate "
        f"({pubmed_best['oov_rate_tokens'] * 100:.2f}%)."
    )


st.info(
    "WordPiece achieves the lowest OOV rate in both domains, "
    "while producing more tokens because it decomposes words "
    "into subword units."
)


# ============================================================
# MANUAL INSPECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Manual Inspection of Problematic Examples</div>',
    unsafe_allow_html=True
)

if examples_df is not None and not examples_df.empty:

    st.write(
        f"Total examples available: **{len(examples_df)}**"
    )

    st.dataframe(
        examples_df,
        use_container_width=True,
        hide_index=True
    )

    # Download button
    examples_csv = examples_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Manual Inspection CSV",
        data=examples_csv,
        file_name="manual_inspection_examples.csv",
        mime="text/csv"
    )

else:

    st.warning(
        "No manual inspection examples are available."
    )


# ============================================================
# DOWNLOAD SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Download Results</div>',
    unsafe_allow_html=True
)

summary_csv = summary_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Complete Summary CSV",
    data=summary_csv,
    file_name="summary_metrics.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#777;">
        <b>Assignment 4 — Text and Sentiment Analysis</b><br>
        Comparative Study of Tokenization Methods on Social Media
        and Biomedical Text
    </div>
    """,
    unsafe_allow_html=True
)
