import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Retail Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Retail Customer Segmentation Dashboard")
st.markdown("Interactive view of customer clusters based on RFM (Recency, Frequency, Monetary) Analysis.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_path = 'outputs/customer_segments.csv'
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df

df = load_data()

if df is None:
    st.error("⚠️ Data not found! Please run `python segmentation.py` first to generate the clustered data.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
all_segments = df['Segment'].unique().tolist()
selected_segments = st.sidebar.multiselect(
    "Select Customer Segments:",
    options=all_segments,
    default=all_segments
)

# Apply filter
filtered_df = df[df['Segment'].isin(selected_segments)]

# --- HIGH-LEVEL METRICS ---
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
total_revenue = filtered_df['Monetary'].sum()
avg_recency = filtered_df['Recency'].mean()
avg_frequency = filtered_df['Frequency'].mean()

col1.metric("Total Customers Selected", f"{total_customers:,}")
col2.metric("Total Revenue ($)", f"${total_revenue:,.2f}")
col3.metric("Avg Recency (Days)", f"{avg_recency:.1f}")
col4.metric("Avg Frequency (Purchases)", f"{avg_frequency:.1f}")

st.divider()

# --- VISUALIZATIONS ---
st.header("Cluster Visualizations")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Recency vs Monetary Value")
    fig_rm = px.scatter(
        filtered_df, 
        x='Recency', 
        y='Monetary', 
        color='Segment',
        hover_data=['CustomerID', 'Frequency'],
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_rm, use_container_width=True)

with col_chart2:
    st.subheader("Frequency vs Monetary Value")
    fig_fm = px.scatter(
        filtered_df, 
        x='Frequency', 
        y='Monetary', 
        color='Segment',
        hover_data=['CustomerID', 'Recency'],
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_fm, use_container_width=True)

# Additional Insights
st.subheader("Segment Distribution")
fig_pie = px.pie(
    filtered_df, 
    names='Segment', 
    title='Proportion of Customers per Segment',
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# --- RAW DATA VIEW ---
st.header("Raw Customer Data")
st.dataframe(filtered_df.sort_values(by='Monetary', ascending=False), use_container_width=True)
