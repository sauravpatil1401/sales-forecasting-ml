import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("Sales Forecasting Dashboard")

df = pd.read_csv("sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

st.subheader("Sales Data")
st.dataframe(df.tail(20), use_container_width=True)

st.subheader("Sales Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["date"], df["sales"])
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Daily Sales Trend")
st.pyplot(fig)

st.subheader("Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total Sales", int(df["sales"].sum()))
c2.metric("Average Daily Sales", round(df["sales"].mean(), 2))
c3.metric("Maximum Daily Sales", int(df["sales"].max()))
