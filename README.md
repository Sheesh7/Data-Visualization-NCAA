# 🏀 HoopsIQ: NCAA Women’s Basketball Advanced Analytics Suite

> **Submission for the Boiler Up, Hammer Data Challenge**

A high-rigor, interactive web application designed to evaluate team performance, offensive efficiency, and game-changing clutch moments across millions of NCAA Women's Basketball play-by-play events.

---

## 📌 Executive Summary

**HoopsIQ** transforms raw play-by-play telemetry into actionable tactical scouting insights for coaches, analysts, and judges. By combining custom Expected Points Added (**EPA**) with a high-accuracy Win Probability Added (**WPA**) logistic regression engine, the dashboard exposes win-volatility trends, situational score-margin execution, and individual clutch performances.

---

## 🚀 Key Features & Analytical Capabilities

* **High-Accuracy Modeling:** Integrates custom Expected Points Added (**EPA**) and Win Probability Added (**WPA**) metrics derived from a logistic regression framework achieving a **ROC-AUC of 0.9072**.
* **Interactive Program Scouting:** Filter macro analytics dynamically by specific teams/schools to analyze individual program execution and volatility under pressure.
* **Win Probability Shift (WPA) Distribution:** Visualizes the frequency and magnitude of game-swinging plays, cleanly separating routine plays in blowouts from high-leverage swings.
* **Game Leverage & Volatility Analysis:** Scatter plot mapping situational game tension against time remaining, featuring rich hover data to surface individual game contexts.
* **Situational Score Margin Breakdown:** Quantifies how team efficiency and shot volume shift across 6 distinct game states (from blowout leads to tight late-game comebacks).
* **Top Clutch Performances Leaderboard:** Surfaces and ranks the top 5 highest absolute WPA plays from the current filter, providing a concrete narrative highlight reel for judges.

---

## 🛠️ Technical Stack & Data Pipeline

* **Frontend / UI:** [Streamlit](https://streamlit.io/) (engineered for sub-second reactive filtering)
* **Data Visualization:** [Plotly Express](https://plotly.com/python/)
* **Data Engineering Stack:** Python, Pandas, NumPy, PyArrow (batch-processing 2M+ raw events down to optimized performance subsets)
* **Machine Learning:** Scikit-Learn (Logistic Regression for Win Probability & Leverage state modeling)

---

## 📊 Analytical Methodology & Rigor

1. **Scalable Data Architecture:** Processed over 2.09 million play-by-play events using PyArrow batching and columnar Parquet storage to maintain sub-second dashboard responsiveness.
2. **Noise Reduction & Statistical Stability:** Applied dynamic sampling and sample-size filtering to eliminate small-sample flukes and ensure macro-level statistical reliability.
3. **Situational Feature Engineering:** Standardized score differentials, remaining time, period states, and possession outcomes to bridge raw play logs with professional front-office evaluation frameworks.

---

## 📦 Local Installation & Running the App

### 1. Clone the Repository
```bash
git clone [https://github.com/Sheesh7/Data-Visualization-NCAA.git](https://github.com/Sheesh7/Data-Visualization-NCAA.git)
cd Data-Visualization-NCAA
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed, then install the required packages:

Bash
pip install streamlit pandas numpy plotly pyarrow

### 3. Run the Streamlit Dashboard
Bash
streamlit run app.py

🏆 Project Context
Developed for the Boiler Up, Hammer Data Challenge to showcase data visualization, statistical modeling rigor, and end-to-end sports analytics engineering.
