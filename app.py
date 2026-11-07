import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="NCAA Women's Basketball Analytics Dashboard",
    page_icon="🏀",
    layout="wide"
)

DATA_DIR = "OneDrive/Documents/Data_Vis_NCAA/data"

@st.cache_data
def load_wpa_data():
    df_path = os.path.join(DATA_DIR, "master_shots_with_wpa.parquet")
    if not os.path.exists(df_path):
        df_path = "master_shots_with_wpa.parquet"
        
    df = pd.read_parquet(df_path)
    
    # Optimized sampling for sub-second UI responsiveness
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
        
    # Feature Engineering & Metric Calculations
    if 'points_attempted' not in df.columns:
        df['points_attempted'] = np.where(df['is_three'] == 1, 3.0, 2.0)
    if 'expected_points' not in df.columns:
        df['expected_points'] = np.where(df['is_three'] == 1, 1.05, 1.10)
    if 'actual_points' not in df.columns:
        df['actual_points'] = df['shot_made'] * df['points_attempted']
    if 'epa' not in df.columns:
        df['epa'] = df['actual_points'] - df['expected_points']
        
    return df

# Load Dataset
df = load_wpa_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Dashboard Controls")

# Shot Category Filter
shot_filter = st.sidebar.selectbox("Filter by Shot Category", ["All", "2-Pointers", "3-Pointers"])

# Team / School Scouting Filter
team_col = None
for col in ['team_name', 'team', 'school', 'home_team']:
    if col in df.columns:
        team_col = col
        break

selected_teams = []
if team_col:
    available_teams = sorted(df[team_col].dropna().unique())
    selected_teams = st.sidebar.multiselect("Filter by Team / School", options=available_teams)

# Apply Filter Logic
filtered_df = df.copy()
if shot_filter == "2-Pointers":
    filtered_df = filtered_df[filtered_df['is_three'] == 0]
elif shot_filter == "3-Pointers":
    filtered_df = filtered_df[filtered_df['is_three'] == 1]

if team_col and selected_teams:
    filtered_df = filtered_df[filtered_df[team_col].isin(selected_teams)]

# --- MAIN HEADER ---
st.title("Boiler Up, Hammer Data Challenge Submission")
st.markdown("Quantifying offensive efficiency (EPA) and game-changing clutch moments (WPA & Leverage) across processed NCAA play-by-play data.")

# --- TOP METRIC SUMMARY CARDS ---
col1, col2, col3, col4 = st.columns(4)
total_shots = len(filtered_df)
fg_pct = filtered_df['shot_made'].mean() * 100 if total_shots > 0 else 0.0
avg_epa = filtered_df['epa'].mean() if total_shots > 0 else 0.0
avg_lev = filtered_df['leverage'].mean() if 'leverage' in filtered_df.columns and total_shots > 0 else 0.0

col1.metric("Shots Analyzed", f"{total_shots:,}")
col2.metric("Field Goal Success Rate", f"{fg_pct:.1f}%")
col3.metric("Avg EPA per Shot", f"{avg_epa:.3f}")
col4.metric("Avg Game Leverage", f"{avg_lev:.3f}")

st.markdown("---")

# --- SECTION 1: WPA DISTRIBUTION & LEVERAGE SCATTER ---
col_wpa, col_lev = st.columns(2)

with col_wpa:
    st.subheader("Win Probability Added (WPA) Distribution")
    if len(filtered_df) > 0 and 'wpa' in filtered_df.columns:
        # Create a clean string label column for coloring the histogram safely
        plot_hist_df = filtered_df.copy()
        plot_hist_df['Shot Result'] = plot_hist_df['shot_made'].map({1: "Made Shot", 0: "Missed Shot"})
        
        fig_wpa = px.histogram(
            plot_hist_df, x="wpa", color="Shot Result",
            nbins=40, title="Shift in Win Probability per Shot",
            labels={"wpa": "Win Probability Added (WPA)", "count": "Frequency"}
        )
        st.plotly_chart(fig_wpa, use_container_width=True)
    else:
        st.info("WPA data unavailable for current selection.")

with col_lev:
    st.subheader("Game Leverage vs. Shot Efficiency")
    if len(filtered_df) > 0 and 'leverage' in filtered_df.columns and 'start_game_seconds_remaining' in filtered_df.columns:
        hover_cols = [c for c in ['team_name', 'opponent', 'score_diff', 'wpa'] if c in filtered_df.columns]
        
        # Plot without continuous colorscale direct conflict on marginals
        fig_leverage = px.scatter(
            filtered_df.head(10000),
            x="start_game_seconds_remaining",
            y="leverage",
            color="wpa",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            hover_data=hover_cols,
            title="Game Leverage Across Time Remaining",
            labels={"start_game_seconds_remaining": "Seconds Remaining in Game", "leverage": "Leverage (Volatility)", "wpa": "WPA"}
        )
        # Add marginal distributions via update layout / traces safely or use standard scatter to avoid marginal bug if needed
        fig_leverage.update_layout(height=450)
        st.plotly_chart(fig_leverage, use_container_width=True)
    else:
        st.info("Leverage data unavailable for current selection.")

# --- SECTION 2: OFFENSIVE EFFICIENCY BY GAME MARGIN ---
st.markdown("---")
st.subheader("📊 Offensive Efficiency by Game Margin (Score Difference)")
st.markdown("Analyzing how shot volume and Expected Points Added (EPA) shift based on game state (Trailing, Close, or Leading).")

if len(filtered_df) > 0 and 'score_diff' in filtered_df.columns:
    bins = [-100, -10, -5, 0, 5, 10, 100]
    labels = ["Trailing Big (>10)", "Trailing (6-10)", "Trailing (1-5)", "Leading (1-5)", "Leading (6-10)", "Leading Big (>10)"]
    
    filtered_df['margin_bucket'] = pd.cut(filtered_df['score_diff'], bins=bins, labels=labels)
    
    margin_stats = filtered_df.groupby('margin_bucket', observed=False).agg(
        total_shots=('shot_made', 'count'),
        avg_epa=('epa', 'mean')
    ).reset_index()
    
    fig_margin = px.bar(
        margin_stats,
        x="margin_bucket",
        y="total_shots",
        color="avg_epa",
        color_continuous_scale="Blues",
        title="Shot Volume and Efficiency by Game Score Margin",
        labels={"margin_bucket": "Score Margin Situation", "total_shots": "Total Shot Volume", "avg_epa": "Avg EPA"}
    )
    fig_margin.update_layout(xaxis_title="Game Situation", yaxis_title="Shot Volume", height=450)
    st.plotly_chart(fig_margin, use_container_width=True)
else:
    st.info("Score difference data unavailable for current selection.")

# --- SECTION 3: TOP CLUTCH PERFORMANCES TABLE ---
st.markdown("---")
st.subheader("⭐ Top 5 Clutch Performances (Highest WPA Plays)")
st.markdown("The highest-impact plays from the current dataset filter, ranked by absolute Win Probability Added (WPA).")

if len(filtered_df) > 0 and 'wpa' in filtered_df.columns:
    clutch_df = filtered_df.reindex(filtered_df['wpa'].abs().sort_values(ascending=False).index).head(5)
    display_columns = [c for c in ['team_name', 'opponent', 'period', 'start_game_seconds_remaining', 'score_diff', 'wpa', 'shot_made'] if c in clutch_df.columns]
    
    if display_columns:
        st.dataframe(clutch_df[display_columns], use_container_width=True)
    else:
        st.dataframe(clutch_df[['wpa']], use_container_width=True)
else:
    st.info("Insufficient data for clutch performance ranking.")
