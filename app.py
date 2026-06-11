
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Air Quality Explorer",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .insight-box {
        background: #eaf4fb;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14px;
        color: #2c3e50;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 6px;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality_with_hri.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_summary():
    return pd.read_csv("city_hri_summary.csv")

df_full    = load_data()
city_hri   = load_summary()

# ── Constants ─────────────────────────────────────────────────
CITY_COORDS = {
    "Delhi"     : [28.6139, 77.2090],
    "Mumbai"    : [19.0760, 72.8777],
    "Chennai"   : [13.0827, 80.2707],
    "Kolkata"   : [22.5726, 88.3639],
    "Bangalore" : [12.9716, 77.5946],
    "Hyderabad" : [17.3850, 78.4867],
    "Ahmedabad" : [23.0225, 72.5714],
    "Lucknow"   : [26.8467, 80.9462],
    "Patna"     : [25.5941, 85.1376],
    "Gurugram"  : [28.4595, 77.0266],
}

CATEGORY_COLORS = {
    "Good"           : "#27ae60",
    "Moderate"       : "#f1c40f",
    "Unhealthy"      : "#e67e22",
    "Very Unhealthy" : "#e74c3c",
    "Hazardous"      : "#8e44ad",
    "Unknown"        : "#bdc3c7",
}

def classify_hri(score):
    if pd.isna(score):    return "Unknown"
    elif score <= 50:     return "Good"
    elif score <= 100:    return "Moderate"
    elif score <= 200:    return "Unhealthy"
    elif score <= 300:    return "Very Unhealthy"
    else:                 return "Hazardous"

def hri_map_color(score):
    if score <= 50:    return "#27ae60"
    elif score <= 100: return "#f1c40f"
    elif score <= 200: return "#e67e22"
    elif score <= 300: return "#e74c3c"
    else:              return "#8e44ad"

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/air-quality.png",
    width=80
)
st.sidebar.title("Air Quality Explorer")
st.sidebar.markdown("---")

all_cities = sorted(df_full["City"].unique().tolist())
selected_cities = st.sidebar.multiselect(
    "Select Cities",
    options=all_cities,
    default=all_cities[:5]
)

min_date = df_full["Date"].min().date()
max_date = df_full["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

selected_pollutant = st.sidebar.selectbox(
    "Focus Pollutant",
    options=["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this project**
Urban Air Quality Explorer analyzes
daily pollution data across 10 Indian
cities (2015–2020) and computes a
custom Health Risk Index (HRI) based
on WHO 2021 guidelines.
""")

# ── Filter data ───────────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df_full[
        (df_full["City"].isin(selected_cities)) &
        (df_full["Date"] >= pd.Timestamp(start_date)) &
        (df_full["Date"] <= pd.Timestamp(end_date))
    ].copy()
else:
    df = df_full[df_full["City"].isin(selected_cities)].copy()

city_summary = city_hri[city_hri["City"].isin(selected_cities)]

# ── Main title ────────────────────────────────────────────────
st.markdown(
    "<p class=main-title>Urban Air Quality Explorer</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class=sub-title>City-wise Pollution Patterns & Health Risk Index — India (2015–2020)</p>",
    unsafe_allow_html=True
)

# ── KPI cards ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

avg_hri    = city_summary["Avg_HRI"].mean()
max_city   = city_summary.loc[city_summary["Avg_HRI"].idxmax(), "City"]
avg_pm25   = df["PM2.5"].mean()
unhealthy  = city_summary["Unhealthy_%"].mean()

k1.metric("Average HRI Score",    f"{avg_hri:.1f}",   classify_hri(avg_hri))
k2.metric("Highest Risk City",    max_city)
k3.metric("Average PM2.5",        f"{avg_pm25:.1f} µg/m³",
          f"WHO limit: 15 µg/m³")
k4.metric("Avg Unhealthy Days",   f"{unhealthy:.1f}%")

st.markdown("---")

# ── Row 1: Map + Radar ────────────────────────────────────────
col_map, col_radar = st.columns([1.2, 1])

with col_map:
    st.markdown("<p class=section-header>City Health Risk Map</p>",
                unsafe_allow_html=True)

    m = folium.Map(location=[22.5, 80.0], zoom_start=4,
                   tiles="CartoDB positron")

    for _, row in city_summary.iterrows():
        if row["City"] not in CITY_COORDS:
            continue
        lat, lon = CITY_COORDS[row["City"]]
        color    = hri_map_color(row["Avg_HRI"])

        folium.CircleMarker(
            location=[lat, lon],
            radius=row["Avg_HRI"] / 10 + 6,
            color=color, fill=True,
            fill_color=color, fill_opacity=0.8,
            weight=2,
            popup=folium.Popup(
                f"<b>{row['City']}</b><br>"
                f"HRI: {row['Avg_HRI']:.1f}<br>"
                f"Risk: {row['Risk_Label']}<br>"
                f"PM2.5: {row['Avg_PM25']:.1f} µg/m³",
                max_width=160
            ),
            tooltip=f"{row['City']}: {row['Avg_HRI']:.0f}"
        ).add_to(m)

        folium.Marker(
            location=[lat + 0.7, lon],
            icon=folium.DivIcon(
                html=f"<div style='font-size:10px;font-weight:bold;"
                     f"color:{color}'>{row['City']}</div>",
                icon_size=(90, 18)
            )
        ).add_to(m)

    st_folium(m, height=380, use_container_width=True)

with col_radar:
    st.markdown("<p class=section-header>Pollutant Profile (Radar)</p>",
                unsafe_allow_html=True)

    radar_cols  = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    radar_data  = df.groupby("City")[radar_cols].mean().reset_index()
    scaler      = MinMaxScaler()
    radar_data[radar_cols] = scaler.fit_transform(radar_data[radar_cols])

    city_colors_list = [
        "#e74c3c","#3498db","#2ecc71","#f39c12",
        "#9b59b6","#1abc9c","#e67e22","#34495e",
        "#e91e63","#00bcd4"
    ]

    fig_radar = go.Figure()
    for i, row in radar_data.iterrows():
        vals = row[radar_cols].tolist() + [row[radar_cols[0]]]
        cats = radar_cols + [radar_cols[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill="toself", name=row["City"],
            fillcolor=city_colors_list[i % len(city_colors_list)],
            opacity=0.15,
            line=dict(color=city_colors_list[i % len(city_colors_list)],
                      width=2)
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=380,
        margin=dict(t=20, b=20, l=20, r=120),
        legend=dict(font=dict(size=10), x=1.0, y=0.5),
        showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Row 2: Time-series ────────────────────────────────────────
st.markdown(
    f"<p class=section-header>{selected_pollutant} Trend Over Time</p>",
    unsafe_allow_html=True
)

monthly = (df.groupby(["City","Year","Month"])[selected_pollutant]
           .mean().reset_index())
monthly["Date"] = pd.to_datetime(
    monthly[["Year","Month"]].assign(DAY=1)
)

fig_ts = px.line(
    monthly.sort_values("Date"),
    x="Date", y=selected_pollutant,
    color="City",
    title="",
    labels={selected_pollutant: f"{selected_pollutant} (µg/m³)",
            "Date": ""},
    color_discrete_sequence=city_colors_list
)
fig_ts.update_layout(
    height=360,
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    margin=dict(t=10, b=40, l=60, r=20),
    legend=dict(orientation="h", y=-0.2)
)
fig_ts.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)")
fig_ts.update_xaxes(showgrid=False,
                    rangeslider=dict(visible=True))
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# ── Row 3: HRI bar + Category dist ───────────────────────────
col_bar, col_cat = st.columns(2)

with col_bar:
    st.markdown("<p class=section-header>City HRI Ranking</p>",
                unsafe_allow_html=True)

    sorted_summary = city_summary.sort_values("Avg_HRI", ascending=False)
    bar_colors = [CATEGORY_COLORS.get(l, "#95a5a6")
                  for l in sorted_summary["Risk_Label"]]

    fig_bar = go.Figure(go.Bar(
        x=sorted_summary["City"],
        y=sorted_summary["Avg_HRI"],
        marker_color=bar_colors,
        marker_line_color="white",
        marker_line_width=1,
        text=[f"{v:.0f}" for v in sorted_summary["Avg_HRI"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>HRI: %{y:.1f}<extra></extra>"
    ))
    fig_bar.add_hline(y=50,  line_dash="dash",
                     line_color="#27ae60", line_width=1)
    fig_bar.add_hline(y=100, line_dash="dash",
                     line_color="#e67e22", line_width=1)
    fig_bar.update_layout(
        height=340, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=40, l=50, r=20),
        yaxis=dict(title="Avg HRI", range=[0, 320])
    )
    fig_bar.update_xaxes(showgrid=False, tickangle=-20)
    fig_bar.update_yaxes(showgrid=True,
                         gridcolor="rgba(0,0,0,0.07)")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_cat:
    st.markdown("<p class=section-header>Season-wise HRI</p>",
                unsafe_allow_html=True)

    df["Season"] = df["Month"].map({
        12:"Winter", 1:"Winter",  2:"Winter",
        3:"Spring",  4:"Spring",  5:"Spring",
        6:"Summer",  7:"Summer",  8:"Summer",
        9:"Monsoon", 10:"Monsoon",11:"Monsoon"
    })

    season_hri = df.groupby(["City","Season"])["HRI"].mean().reset_index()

    fig_season = px.bar(
        season_hri,
        x="Season", y="HRI",
        color="City",
        barmode="group",
        category_orders={"Season":["Winter","Spring",
                                    "Summer","Monsoon"]},
        color_discrete_sequence=city_colors_list,
        labels={"HRI": "Average HRI"},
        title=""
    )
    fig_season.update_layout(
        height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=40, l=50, r=20),
        legend=dict(font=dict(size=9),
                    orientation="h", y=-0.3)
    )
    fig_season.update_yaxes(showgrid=True,
                            gridcolor="rgba(0,0,0,0.07)")
    st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")

# ── Key insights ──────────────────────────────────────────────
st.markdown("<p class=section-header>Key Insights</p>",
            unsafe_allow_html=True)

insights = [
    "Delhi and Patna consistently show HRI scores 4–5x above the WHO safe threshold, driven primarily by PM2.5 in winter months.",
    "Winter season produces 2–4x higher HRI than Monsoon across all cities — temperature inversion traps pollutants near ground level.",
    "PM2.5 and PM10 show a correlation of ~0.85+, indicating shared emission sources such as vehicles and construction dust.",
    "Bangalore and Chennai remain the safest cities year-round, rarely exceeding the Moderate HRI threshold.",
    "All 10 cities exceed the WHO annual PM2.5 limit of 15 µg/m³, highlighting a systemic national air quality crisis."
]

for insight in insights:
    st.markdown(
        f"<div class=insight-box>• {insight}</div>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<center style='color:#95a5a6; font-size:12px'>"
    "Urban Air Quality Explorer | Data: Kaggle CPCB India AQI | "
    "Built with Python, Streamlit, Plotly & Folium"
    "</center>",
    unsafe_allow_html=True
)
