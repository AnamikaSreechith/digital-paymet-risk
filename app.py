import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digital Payment Risk | Fraud Detection",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #101426;
    color: #F4F5FC;
}

.block-container {
    padding: 2.5rem 4rem 3rem 4rem;
    max-width: 1500px;
}

.hero {
    padding: 1.5rem 0 2rem 0;
}

.brand {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #8B8CFF, #5BC0EB);
    color: #101426;
    font-size: 17px;
    font-weight: 800;
    margin-right: 13px;
    vertical-align: middle;
}

.hero h1 {
    color: #F5F5FF;
    font-size: 43px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
    line-height: 1.2;
}

.section-title {
    color: #F5F5FF;
    font-size: 25px;
    font-weight: 700;
    margin-top: 1.8rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: #1B2038;
    border: 1px solid #303753;
    border-radius: 15px;
    padding: 22px 24px;
    min-height: 115px;
}

.metric-label {
    color: #AEB6D2;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #F5F5FF;
    font-size: 31px;
    font-weight: 800;
}

.metric-note {
    color: #929BB9;
    font-size: 12px;
    margin-top: 5px;
}

.panel {
    background: #1B2038;
    border: 1px solid #303753;
    border-radius: 16px;
    padding: 20px;
}

div[data-testid="stTextInput"] input {
    background: #1B2038;
    color: #FFFFFF;
    border: 1px solid #424B6B;
    border-radius: 10px;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSlider"] {
    color: #FFFFFF;
}

.stButton button, .stDownloadButton button {
    background: #898BFF;
    color: #101426;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #A5A6FF;
    color: #101426;
    border: none;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #303753;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #303753;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL AND SCALER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_xgboost_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "standard_scaler.pkl"

THRESHOLD = 0.97

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>
        <span class="brand">DP</span>
        Digital Payment Risk & Fraud Detection
    </h1>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL OVERVIEW
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Detection Model</div>
        <div class="metric-value">XGBoost</div>
        <div class="metric-note">Gradient boosting classifier</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Decision Threshold</div>
        <div class="metric-value">{THRESHOLD:.2f}</div>
        <div class="metric-note">Fraud classification cutoff</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Input Features</div>
        <div class="metric-value">30</div>
        <div class="metric-note">Transaction attributes</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# UPLOAD DATA
# =========================================================

st.markdown('<div class="section-title">Transaction Analysis</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload transaction dataset",
    type=["csv"],
    help="Upload a CSV containing Time, V1–V28 and Amount."
)

if uploaded_file is None:
    st.info("Upload a transaction CSV to view fraud detection results.")
    st.stop()

# =========================================================
# DATA VALIDATION AND PREDICTION
# =========================================================

try:
    df = pd.read_csv(uploaded_file)

    feature_columns = (
        ["Time"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount"]
    )

    missing_columns = [
        col for col in feature_columns if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    if df.empty:
        st.error("The uploaded file contains no transactions.")
        st.stop()

    X = df[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        st.error("Missing or non-numeric values found in the feature columns.")
        st.stop()

    # Scale only Time and Amount
    X[["Time", "Amount"]] = scaler.transform(
        X[["Time", "Amount"]]
    )

    # Predict fraud probabilities
    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    results = df.copy()
    results.insert(0, "Transaction_ID",
                   [f"TXN-{i:06d}" for i in range(1, len(df) + 1)])

    results["Fraud_Probability"] = probabilities
    results["Risk_Level"] = np.where(
        probabilities >= THRESHOLD,
        "High Risk",
        np.where(probabilities >= 0.50, "Medium Risk", "Low Risk")
    )
    results["Prediction"] = np.where(
        predictions == 1, "Fraudulent", "Genuine"
    )

    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    total = len(results)
    fraud_count = int(predictions.sum())
    genuine_count = total - fraud_count
    fraud_rate = (fraud_count / total * 100) if total else 0

    st.markdown('<div class="section-title">Detection Overview</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        (m1, "Total Transactions", f"{total:,}", "Transactions analyzed"),
        (m2, "Fraudulent Transactions", f"{fraud_count:,}", "Flagged as high risk"),
        (m3, "Genuine Transactions", f"{genuine_count:,}", "Below fraud threshold"),
        (m4, "Fraud Detection Rate", f"{fraud_rate:.2f}%", "Share flagged as fraud")
    ]

    for col, label, value, note in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # VISUAL ANALYTICS
    # =====================================================

    st.markdown('<div class="section-title">Risk Analytics</div>',
                unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        pie_data = pd.DataFrame({
            "Transaction Status": ["Genuine", "Fraudulent"],
            "Count": [genuine_count, fraud_count]
        })

        fig_pie = px.pie(
            pie_data,
            names="Transaction Status",
            values="Count",
            hole=0.68,
            color="Transaction Status",
            color_discrete_map={
                "Genuine": "#65C7A5",
                "Fraudulent": "#FF718B"
            }
        )

        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent",
            hole=0.68
        )

        fig_pie.update_layout(
            title="Transaction Classification",
            title_font_size=18,
            title_font_color="#F5F5FF",
            paper_bgcolor="#1B2038",
            plot_bgcolor="#1B2038",
            font_color="#E9ECFA",
            legend_title_text="",
            margin=dict(t=65, b=25, l=20, r=20),
            height=370
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with chart2:
        risk_data = results["Risk_Level"].value_counts().reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        ).reset_index()

        risk_data.columns = ["Risk Level", "Transactions"]

        fig_bar = px.bar(
            risk_data,
            x="Risk Level",
            y="Transactions",
            color="Risk Level",
            text="Transactions",
            color_discrete_map={
                "Low Risk": "#65C7A5",
                "Medium Risk": "#F6C66B",
                "High Risk": "#FF718B"
            }
        )

        fig_bar.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        fig_bar.update_layout(
            title="Transaction Risk Distribution",
            title_font_size=18,
            title_font_color="#F5F5FF",
            paper_bgcolor="#1B2038",
            plot_bgcolor="#1B2038",
            font_color="#E9ECFA",
            showlegend=False,
            yaxis_title="Number of Transactions",
            xaxis_title="",
            margin=dict(t=65, b=30, l=20, r=20),
            height=370
        )

        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(gridcolor="#343B57")

        st.plotly_chart(fig_bar, use_container_width=True)

    # =====================================================
    # SEARCH AND FILTER TRANSACTIONS
    # =====================================================

    st.markdown('<div class="section-title">Explore Transactions</div>',
                unsafe_allow_html=True)

    search_col, status_col = st.columns([2, 1])

    with search_col:
        search_query = st.text_input(
            "Search transactions",
            placeholder="Search Transaction ID, amount or status..."
        )

    with status_col:
        status_filter = st.selectbox(
            "Transaction Status",
            ["All", "Genuine", "Fraudulent"]
        )

    filtered = results.copy()

    if search_query:
        query = search_query.lower()
        mask = (
            filtered["Transaction_ID"].str.lower().str.contains(query, na=False)
            | filtered["Prediction"].str.lower().str.contains(query, na=False)
            | filtered["Risk_Level"].str.lower().str.contains(query, na=False)
            | filtered["Amount"].astype(str).str.contains(query, na=False)
            | filtered["Time"].astype(str).str.contains(query, na=False)
        )
        filtered = filtered[mask]

    if status_filter != "All":
        filtered = filtered[
            filtered["Prediction"] == status_filter
        ]


    # =====================================================
    # PAGINATED TRANSACTION TABLE
    # =====================================================

    st.caption(
        f"Showing {len(filtered):,} matching transactions "
        f"out of {total:,} total transactions"
    )

    page_size = st.selectbox(
        "Rows per page",
        [10, 25, 50, 100],
        index=1,
        key="transaction_page_size"
    )

    total_filtered = len(filtered)
    total_pages = max(1, (total_filtered + page_size - 1) // page_size)

    # Reset pagination whenever the search or filters change
    filter_signature = (
        search_query,
        status_filter,
        page_size
    )

    if st.session_state.get("previous_filter_signature") != filter_signature:
        st.session_state.transaction_page = 1
        st.session_state.previous_filter_signature = filter_signature

    current_page = st.session_state.get("transaction_page", 1)
    current_page = min(max(current_page, 1), total_pages)
    st.session_state.transaction_page = current_page

    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size

    display_columns = [
        "Transaction_ID",
        "Time",
        "Amount",
        "Fraud_Probability",
        "Risk_Level",
        "Prediction"
    ]

    page_data = filtered.iloc[start_index:end_index][display_columns].copy()

    # Convert probability to percentage for clear table display
    page_data["Fraud_Probability"] = (
        page_data["Fraud_Probability"] * 100
    )

    st.dataframe(
        page_data,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Transaction_ID": st.column_config.TextColumn(
                "Transaction ID"
            ),
            "Time": st.column_config.NumberColumn(
                "Time",
                format="%.0f"
            ),
            "Amount": st.column_config.NumberColumn(
                "Amount",
                format="%.2f"
            ),
            "Fraud_Probability": st.column_config.NumberColumn(
                "Fraud Probability",
                format="%.2f%%"
            ),
            "Risk_Level": st.column_config.TextColumn(
                "Risk Level"
            ),
            "Prediction": st.column_config.TextColumn(
                "Prediction"
            )
        }
    )

    # Pagination navigation
    previous_col, page_info_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button(
            "← Previous",
            disabled=(current_page <= 1),
            use_container_width=True,
            key="previous_transaction_page"
        ):
            st.session_state.transaction_page = current_page - 1
            st.rerun()

    with page_info_col:
        st.markdown(
            f"<div style='text-align:center; padding:8px 0;'>"
            f"Page <b>{current_page:,}</b> of <b>{total_pages:,}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

    with next_col:
        if st.button(
            "Next →",
            disabled=(current_page >= total_pages),
            use_container_width=True,
            key="next_transaction_page"
        ):
            st.session_state.transaction_page = current_page + 1
            st.rerun()

    # =====================================================
    # SELECT A TRANSACTION
    # =====================================================

    st.markdown('<div class="section-title">Transaction Details</div>',
                unsafe_allow_html=True)

    if not page_data.empty:
        selected_id = st.selectbox(
            "Select a transaction from the current page",
            page_data["Transaction_ID"].tolist(),
            key="selected_transaction_id"
        )

        selected = results[
            results["Transaction_ID"] == selected_id
        ].iloc[0]

        d1, d2, d3 = st.columns(3)

        d1.metric("Transaction Amount", f"{selected['Amount']:.2f}")
        d2.metric("Fraud Probability",
                  f"{selected['Fraud_Probability']:.2%}")
        d3.metric("Prediction", selected["Prediction"])

        if selected["Prediction"] == "Fraudulent":
            st.error("This transaction has been flagged as potentially fraudulent.")
        else:
            st.success("This transaction is classified as genuine by the model.")

        # =====================================================
        # INTERACTIVE, USER-FRIENDLY SHAP EXPLANATION
        # =====================================================

        with st.expander("Why did the model make this decision? — View explanation", expanded=False):
            st.caption(
                "Explore the strongest signals that influenced this transaction's "
                "prediction. The chart updates when you select a different transaction."
            )

            try:
                selected_index = results.index[
                    results["Transaction_ID"] == selected_id
                ][0]
                selected_model_input = X.loc[[selected_index]]

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(selected_model_input)
                base_values = explainer.expected_value

                # Handle SHAP output formats across supported SHAP versions.
                if isinstance(shap_values, list):
                    contributions = np.asarray(shap_values[-1])[0]
                    base_value = (
                        np.asarray(base_values[-1]).item()
                        if isinstance(base_values, (list, np.ndarray))
                        else float(base_values)
                    )
                else:
                    shap_array = np.asarray(shap_values)
                    if shap_array.ndim == 3:
                        contributions = shap_array[0, :, -1]
                    else:
                        contributions = shap_array[0]
                    base_array = np.asarray(base_values)
                    base_value = (
                        float(base_array[-1]) if base_array.ndim > 0
                        else float(base_array)
                    )

                shap_table = pd.DataFrame({
                    "Feature": selected_model_input.columns,
                    "Impact": contributions
                })

                fraud_signals = (
                    shap_table[shap_table["Impact"] > 0]
                    .nlargest(5, "Impact")
                )
                genuine_signals = (
                    shap_table[shap_table["Impact"] < 0]
                    .nsmallest(5, "Impact")
                )
                top_signals = pd.concat(
                    [fraud_signals, genuine_signals]
                ).drop_duplicates(subset="Feature")

                if top_signals.empty:
                    st.info("No individual feature contributions were returned for this transaction.")
                else:
                    top_signals = top_signals.sort_values("Impact")
                    bar_colors = [
                        "#FF718B" if value > 0 else "#65C7A5"
                        for value in top_signals["Impact"]
                    ]

                    fig = go.Figure(go.Bar(
                        x=top_signals["Impact"],
                        y=top_signals["Feature"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{value:+.3f}" for value in top_signals["Impact"]],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            "<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>"
                        )
                    ))
                    fig.add_vline(x=0, line_color="#AEB6D2", line_width=1)
                    fig.update_layout(
                        title="Top factors influencing this prediction",
                        xaxis_title="Model contribution (SHAP value)",
                        yaxis_title="",
                        paper_bgcolor="#101426",
                        plot_bgcolor="#1B2038",
                        font=dict(color="#F4F5FC"),
                        xaxis=dict(gridcolor="#303753", zeroline=False),
                        yaxis=dict(automargin=True),
                        margin=dict(l=15, r=45, t=65, b=55),
                        height=390,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fraud_col, genuine_col = st.columns(2)
                    with fraud_col:
                        st.markdown("**↑ Signals pushing towards fraud**")
                        if fraud_signals.empty:
                            st.write("No displayed signals pushed the score towards fraud.")
                        else:
                            for _, item in fraud_signals.iterrows():
                                st.write(f"• **{item['Feature']}** — increased the model's fraud output")

                    with genuine_col:
                        st.markdown("**↓ Signals pushing away from fraud**")
                        if genuine_signals.empty:
                            st.write("No displayed signals pushed the score away from fraud.")
                        else:
                            for _, item in genuine_signals.iterrows():
                                st.write(f"• **{item['Feature']}** — decreased the model's fraud output")

                    st.info(
                        "How to read this: pink bars increase the model's fraud score; "
                        "green bars decrease it. Longer bars indicate a stronger influence. "
                        "V1–V28 are anonymized PCA components, so they cannot be translated "
                        "into real-world descriptions such as merchant, location or device. "
                        "SHAP values are model contributions, not percentages or proof of fraud."
                    )

            except Exception as shap_error:
                st.warning(f"The explanation could not be generated for this transaction: {shap_error}")

        with st.expander("View transaction information"):
            detail_columns = [
                "Transaction_ID", "Time", "Amount",
                "Fraud_Probability", "Risk_Level", "Prediction"
            ]
            detail = selected[detail_columns].to_frame().T.copy()
            detail = detail.rename(columns={
                "Transaction_ID": "Transaction ID",
                "Fraud_Probability": "Fraud Probability",
                "Risk_Level": "Risk Level"
            })
            detail["Fraud Probability"] = (
                detail["Fraud Probability"].astype(float) * 100
            ).map(lambda value: f"{value:.2f}%")
            detail["Amount"] = detail["Amount"].astype(float).map(
                lambda value: f"{value:.2f}"
            )
            st.dataframe(detail, use_container_width=True, hide_index=True)
    else:
        st.warning("No transactions match the selected search and filters.")

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.markdown('<div class="section-title">Export Results</div>',
                unsafe_allow_html=True)

    # Export only fields that are useful to a human reader.
    # V1-V28 are anonymized PCA components, so they are intentionally
    # excluded from both downloadable reports.
    export_columns = [
        "Transaction_ID", "Time", "Amount",
        "Fraud_Probability", "Risk_Level", "Prediction"
    ]

    def prepare_export(dataframe):
        export_df = dataframe[export_columns].copy()
        export_df = export_df.rename(columns={
            "Transaction_ID": "Transaction ID",
            "Fraud_Probability": "Fraud Probability",
            "Risk_Level": "Risk Level"
        })
        export_df["Fraud Probability"] = (
            export_df["Fraud Probability"] * 100
        ).round(2)
        export_df["Amount"] = export_df["Amount"].round(2)
        return export_df

    full_export = prepare_export(results)
    filtered_export = prepare_export(filtered)

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "Download Full Prediction Report",
            data=full_export.to_csv(index=False).encode("utf-8"),
            file_name="full_fraud_prediction_report.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_full_report"
        )

    with download_col2:
        st.download_button(
            "Download Filtered Transactions",
            data=filtered_export.to_csv(index=False).encode("utf-8"),
            file_name="filtered_fraud_transactions.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=filtered_export.empty,
            key="download_filtered_report"
        )

    st.caption(
        f"Full report: {len(full_export):,} transactions | "
        f"Filtered report: {len(filtered_export):,} transactions"
    )
    st.caption("Digital Payment Risk & Fraud Detection | XGBoost")

except Exception as e:
    st.error(f"Unable to process the uploaded dataset: {e}")
