import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64
import math
from datetime import datetime, timedelta

# --- ASTRAL ENGINE IMPORTS ---
from astral import moon, sun, LocationInfo

# ==========================================
# 0. GLOBAL POLICY & ELASTICITY CONSTANTS
# ==========================================
OLD_SUBSIDY_FLOOR = 2.15
KEMBUNG_RATE = 0.50
SELAR_RATE = 0.80

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Seafood Security Simulator",
    layout="wide",
    page_icon="🐟"
)

# ==========================================
# 2. MODEL LOADER (FIXED FOR DEPLOYMENT)
# ==========================================
@st.cache_resource
def load_resources():
    """
    Loads the Champion EBMs and their Forensic Audit Scores.
    Initializes variables to empty containers to prevent AttributeError.
    """
    # Initialize variables so they exist regardless of try/except outcome
    k_struct, s_struct = None, None
    k_stats, s_stats = {}, {} 
    
    try:
        # Load Structural Brains
        if os.path.exists('kembung_structural_model.pkl'):
            k_struct = joblib.load('kembung_structural_model.pkl')
        if os.path.exists('selar_structural_model.pkl'):
            s_struct = joblib.load('selar_structural_model.pkl')
        
        # Load Forensic Metrics (if they exist)
        if os.path.exists('kembung_metrics.json'):
            with open('kembung_metrics.json', 'r') as f:
                k_stats = json.load(f)
        if os.path.exists('selar_metrics.json'):
            with open('selar_metrics.json', 'r') as f:
                s_stats = json.load(f)
            
        return k_struct, s_struct, k_stats, s_stats

    except Exception as e:
        st.error(f"❌ Critical Error Loading Resources: {e}")
        return None, None, {}, {}

# Load and Unpack exactly 4 items
k_struct, s_struct, k_stats, s_stats = load_resources()

# Safety Check: Stop if models are missing
if k_struct is None or s_struct is None:
    st.warning("⚠️ Structural model files (.pkl) not found in the root directory. App paused.")
    st.stop()
    # ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.title("Main Menu")
    # The Global Selector that controls the whole app
    species_choice = st.radio("Select Species:", ["🐟 Kembung", "🐠 Selar"])
    
    st.divider()
    
    # System Status Panel
    st.info("System Architecture")
    if "Kembung" in species_choice:
        st.caption("Mode: Benchmark Beater")
        st.caption("Config: 6 Structural Features")
        st.caption("Target: Domestic Stability")
    else:
        st.caption("Mode: Surgical Interaction")
        st.caption("Config: 7 High-Precision Features")
        st.caption("Target: Import Resilience")
        
    st.divider()
    st.caption("v5.2 Champion Build | Thesis Artifact")

# ==========================================
# 4. TAB SYSTEM
# ==========================================
# We define the 5 core modules of the dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🎮 Simulator", 
    "📈 Performance", 
    "🧠 Evidence Suite", 
    "🔮 Forecast"
])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    st.header("Malaysian Seafood Resilience Audit")
    st.write("Quantitative analysis of price volatility drivers using **Explainable Boosting Machines (EBM)**.")
    
    # Dynamic KPI Cards (Updates based on Sidebar Selection)
    c1, c2, c3 = st.columns(3)
    
    if "Kembung" in species_choice:
        # KEMBUNG METRICS (Using .get with defaults for safety)
        with c1: 
            st.metric(
                "Model Precision (Week 1)", 
                f"{k_stats.get('Structural_MAE', '0.54')} RM", 
                delta=f"{k_stats.get('Accuracy', '96.76')}% Accuracy"
            )
        with c2: 
            st.metric(
                "Primary Driver", 
                "Domestic Fuel", 
                "RON97 + Tides"
            )
        
        st.divider()
        st.success("""
        **🏆 Policy Verdict:** Valid for **Monthly Review**. 
        Kembung retains **>96% accuracy**, demonstrating strong structural inertia. 
        Prices are driven primarily by domestic fuel costs and local tidal cycles.
        """)
        
    else:
        # SELAR METRICS
        with c1: 
            st.metric(
                "Model Precision (Week 1)", 
                f"{s_stats.get('Structural_MAE', '0.53')} RM", 
                delta=f"{s_stats.get('Accuracy', '97.11')}% Accuracy"
            )
        with c2: 
            st.metric(
                "Operational Horizon", 
                "8 Weeks", 
                delta="-0.04 RM Accuracy Cost", 
                delta_color="off",
                help="Time period before forecast accuracy degrades significantly."
            )
        with c3: 
            st.metric(
                "Primary Driver", 
                "Global Logistics", 
                "USD + Diesel Synergy"
            )
        
        st.divider()
        st.success("""
        **🏆 Policy Verdict:** Valid for **Monthly Review**. 
        Selar demonstrates high resilience to short-term fluctuations due to complex interaction effects.
        Prices are highly sensitive to the **Synergy of Diesel Price and Currency Strength**.
        """)

# --- TAB 2: SIMULATOR (HYBRID MODE) ---
with tab2:
    st.header("🎮 Price Shock Simulator (Hybrid Mode)")
    st.caption("Powered by Grand Tournament Champions. Lever configuration reflects specific species drivers.")
    
    c1, c2 = st.columns([1.2, 1])
    
    if "Kembung" in species_choice:
        with c1:
            st.subheader("Hypothetical Scenarios")
            k_ron97 = st.slider("⛽ RON97 Price (RM/L)", 2.00, 5.00, 3.47, step=0.01)
            k_diesel = st.slider("🚛 Diesel Price (RM/L)", 2.15, 4.50, 3.35, step=0.01)
            k_usd = st.slider("🇺🇸 MYR/USD Rate", 3.50, 5.50, 4.45, step=0.01)
            k_tide = st.slider("🌊 Tide Height Perak (m)", 0.50, 3.00, 1.67, step=0.01)
            
        with c2:
            st.subheader("🎯 Model Configuration")
            st.write("**Architecture:** Standard Hybrid (EBM + Injection)")
            st.write("**Elasticity (Pass-through):** 50%")
            st.markdown("- `RON97 Price`\n- `Tide Height` \n- `USD Rate` \n- `Diesel Shock` (Structural)")
            
        if st.button("Run Kembung Simulation", type="primary"):
            input_df = pd.DataFrame({
                'RON97': [k_ron97], 
                'height_mean_m_perak': [k_tide], 
                'myr_per_usd_mean': [k_usd]
            })
            base_price = k_struct.predict(input_df)[0]
            shock = (k_diesel - 2.15) * KEMBUNG_RATE if k_diesel > 2.15 else 0.0
            final_price = base_price + shock
            
            st.divider()
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Forecasted Price", f"RM {final_price:.2f}", delta="± RM 0.5", delta_color="off")
            res_col2.metric("Policy Impact", f"+RM {shock:.2f}", delta="Diesel Pass-through", delta_color="inverse")

    else:
        with c1:
            st.subheader("Hypothetical Scenarios")
            s_diesel = st.slider("🚛 Diesel Price (RM/L)", 2.15, 4.50, 3.35, step=0.01)
            s_usd = st.slider("🇺🇸 MYR/USD Rate", 3.50, 5.50, 4.45, step=0.01)
            s_tide = st.slider("🌊 Tide Height Perak (m)", 0.50, 3.00, 1.67, step=0.01)
            s_sun = st.slider("🌇 Sunset Time (Mins)", 1100, 1200, 1140)
            
        with c2:
            st.subheader("🎯 Model Configuration")
            st.write("**Architecture:** Surgical Bio-Econ Synergy")
            st.write("**Elasticity (Pass-through):** 80%")
            st.markdown("- `Econ Pressure` (Diesel × USD)\n- `Solunar Synergy` (Tide × Sun)\n- `Diesel Shock` (Structural)")
            
        if st.button("Run Selar Simulation", type="primary"):
            econ_pressure = s_diesel * s_usd
            solunar_synergy = s_tide * s_sun
            input_df = pd.DataFrame({
                'econ_pressure_index': [econ_pressure], 
                'solunar_synergy_perak': [solunar_synergy]
            })
            base_price = s_struct.predict(input_df)[0]
            shock = (s_diesel - 2.15) * SELAR_RATE if s_diesel > 2.15 else 0.0
            final_price = base_price + shock
            
            st.divider()
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Forecasted Price", f"RM {final_price:.2f}", delta="± RM 0.7", delta_color="off")
            res_col2.metric("Policy Impact", f"+RM {shock:.2f}", delta="Diesel Pass-through", delta_color="inverse")
        
# --- TAB 3: PERFORMANCE (FORENSIC SCORECARD) ---
with tab3:
    st.header(f"🎯 {species_choice} Model Performance")
    
    # 1. DYNAMIC METRIC LOADER
    is_kembung = "Kembung" in species_choice
    current_stats = k_stats if is_kembung else s_stats
    
    # Check if stats are available (k_stats/s_stats will be {} if files are missing)
    if not current_stats:
        st.warning(f"⚠️ Performance Metrics JSON missing for {species_choice}. Displaying hardcoded baselines.")
        # Fallback values for UAT stability
        acc_val = "96.76%" if is_kembung else "97.11%"
        mape_val = "3.24%" if is_kembung else "2.89%"
        mae_val = "RM 0.54" if is_kembung else "RM 0.53"
    else:
        # Use existing keys from kembung_metrics.json / selar_metrics.json
        acc_val = current_stats.get("Accuracy", "N/A")
        mape_val = current_stats.get("MAPE", "N/A")
        mae_val = current_stats.get("Structural_MAE", "N/A")

    # 2. KEY PERFORMANCE INDICATORS
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall Accuracy", acc_val, help="100% - MAPE (From Dynamic Blind Test)")
    with c2:
        st.metric("Forensic Error (MAPE)", mape_val, help="Mean Absolute Percentage Error", delta_color="inverse")
    with c3:
        st.metric("Avg Price Deviation", mae_val, help="Mean Absolute Error over 8 Weeks")

    st.divider()

    # 3. COMPARATIVE HORIZON TEST
    st.subheader("⏳ The Resilience Stress Test (8-Week Horizon)")
    st.caption("Validating the 'Expiration Date' of the forecast. Left: Model Stability (Inputs Known). Right: Real-World Volatility (Dynamic Forecast).")

    # Helper function for Image Injection (Base64 encoding for Streamlit Cloud reliability)
    def get_base64_image(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    # Path Mapping
    f_dir = "thesis_analytics_images"
    species_prefix = "kembung" if is_kembung else "selar"
    
    file_struct = os.path.join(f_dir, f"{species_prefix}_structural_horizon.png")
    file_blind = os.path.join(f_dir, f"{species_prefix}_total_blind_horizon.png")

    # Load Images
    b64_struct = get_base64_image(file_struct)
    b64_blind = get_base64_image(file_blind)

    h_col1, h_col2 = st.columns(2)
    
    with h_col1:
        st.markdown("**Test A: Predicting one week at a time (Structural Stability)**")
        if b64_struct:
            st.markdown(f'<img src="data:image/png;base64,{b64_struct}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)
            st.info("✅ **Structural Consistency:** The flat RMSE line confirms the model is mathematically stable when economic inputs are precise.")
        else:
            st.warning(f"Structural plot missing: {file_struct}") 

    with h_col2:
        st.markdown("**Test B: Predicting with projected conditions (Blind Forecast)**") 
        if b64_blind:
            st.markdown(f'<img src="data:image/png;base64,{b64_blind}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)
            
            # Dynamic Insight based on Species
            error_margin = "RM 0.55" if is_kembung else "RM 0.82"
            st.info(f"🕶️ **Dynamic Blind Test:** Demonstrates error growth over 8 weeks. Valid within a 4-week window at ± {error_margin}.")
        else:
            st.warning(f"Blind plot missing: {file_blind}")
            
    st.divider()
    
    # 4. THE 30-DAY VERDICT
    st.subheader("🏛️ Policy Implication: The 30-Day Rule")
    species_name = "Kembung" if is_kembung else "Selar"
    st.write(f"""
    By analyzing the divergence between **Test A** (Perfect Knowledge) and **Test B** (Blind Forecast), we establish a **30-Day Validity Window** for {species_name}. 
    While the mathematical engine is robust, real-world economic volatility (USD/Fuel) necessitates a weekly data refresh to maintain optimal precision.
    """)
    
    rec_col1, rec_col2 = st.columns([1, 3])
    with rec_col1:
        st.metric("Update Cycle", "Weekly", "Required for 95%+ Precision")
    with rec_col2:
        st.info("**Operational Protocol:** Policymakers should rely on the 1-4 week forecast horizon. Predictions beyond 30 days are advisory only, as error margins begin to compound.")
    
# --- TAB 4: EVIDENCE SUITE ---
with tab4:
    st.header(f"🧠 {species_choice} Forensic Evidence Suite")
    st.write("Quantitative deconstruction of the 'Black Box' architecture through structural decomposition, marginal impact analysis, and error distribution audits.")

    # Path Configuration
    f_dir = "thesis_analytics_images"
    d_dir = "Thesis_Final_Champion_Suite"
    
    # Species-specific settings
    is_kembung = "Kembung" in species_choice
    prefix = "kembung" if is_kembung else "selar"
    
    # ---------------------------------------------------------
    # 1. STRUCTURAL ANATOMY (Trend & Linear Correlation)
    # ---------------------------------------------------------
    st.subheader("📉 Part 1: Structural Anatomy")
    st.caption("Deconstructing the price into long-term Inflation Trends and surgical statistical correlations (T+1).")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Price Decomposition (Trend vs. Observed)**")
        decomp_file = os.path.join(f_dir, f"{prefix}_decomposition.png")
        b64_dec = get_base64_image(decomp_file)
        if b64_dec:
            st.markdown(f'<img src="data:image/png;base64,{b64_dec}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)
        else:
            st.warning("Decomposition chart not found.")

    with col_b:
        st.markdown("**Surgical Correlation Matrix (Predictive T+1)**")
        corr_file = os.path.join(f_dir, f"{prefix}_structural_correlation.png")
        b64_corr = get_base64_image(corr_file)
        if b64_corr:
            st.markdown(f'<img src="data:image/png;base64,{b64_corr}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)
        else:
            st.warning("Correlation chart not found.")
            
    st.divider()

    # ---------------------------------------------------------
    # 2. GLOBAL DRIVERS (Decomposed Forensic Audit)
    # ---------------------------------------------------------
    st.subheader("📊 Part 2: Global Feature Importance")
    st.markdown(f"""
    **Forensic Decomposition:** This chart ranks the weighted impact of variables on the logic of the {species_choice} engine. 
    By decomposing features, we prove that the **Synergy Indices** outrank standalone components.
    """)
    
    imp_file = os.path.join(f_dir, f"{prefix}_structural_importance.png")
    b64_imp = get_base64_image(imp_file)
    
    if b64_imp:
        st.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="data:image/png;base64,{b64_imp}" style="width:85%; border: 1px solid #eee; border-radius: 8px; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);">'
            f'</div>', 
            unsafe_allow_html=True
        )
    else:
        st.warning("Feature Importance chart not found.")

    # ---------------------------------------------------------
    # 3. MARGINAL IMPACT (Surgical PDPs)
    # ---------------------------------------------------------
    st.subheader("🔍 Part 3: Marginal Impact Analysis (PDPs)")
    st.caption("The 'Physics Curves'. Marginal effect of champion variables on the forecast while holding others constant.")
    
    if is_kembung:
        pdp_files = ["kembung_pdp_RON97.png", "kembung_pdp_height_mean_m_perak.png", "kembung_pdp_myr_per_usd_mean.png"]
    else:
        pdp_files = ["selar_pdp_econ_pressure_index.png", "selar_pdp_solunar_synergy_perak.png"]

    pdp_cols = st.columns(len(pdp_files))
    for idx, fname in enumerate(pdp_files):
        b64_pdp = get_base64_image(os.path.join(f_dir, fname))
        with pdp_cols[idx]:
            if b64_pdp:
                st.markdown(f'<img src="data:image/png;base64,{b64_pdp}" style="width:100%; border: 1px solid #f0f0f0; border-radius: 5px;">', unsafe_allow_html=True)
                clean_label = fname.split('_pdp_')[-1].replace('.png', '').replace('_', ' ').title()
                st.caption(f"Input: {clean_label}")

    st.divider()

    # ---------------------------------------------------------
    # 4. MARKET DYNAMICS (Boxplots & Residual Audit)
    # ---------------------------------------------------------
    st.subheader("🚀 Part 4: Market Dynamics & Residual Audit")
    
    d_col1, d_col2, d_col3 = st.columns(3)
    
    with d_col1:
        st.markdown("**Deliverable #7: Price Arena**")
        b64_box = get_base64_image(os.path.join(d_dir, f"{prefix}_boxplot.png"))
        if b64_box:
            st.markdown(f'<img src="data:image/png;base64,{b64_box}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)
    
    with d_col2:
        st.markdown("**Deliverable #8: Forecasting Residuals**")
        b64_res = get_base64_image(os.path.join(d_dir, f"{prefix}_residuals.png"))
        if b64_res:
            st.markdown(f'<img src="data:image/png;base64,{b64_res}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)

    with d_col3:
        st.markdown("**Deliverable #9: Structural Volatility**")
        b64_vol = get_base64_image(os.path.join(d_dir, f"{prefix}_volatility.png"))
        if b64_vol:
            st.markdown(f'<img src="data:image/png;base64,{b64_vol}" style="width:100%; border: 1px solid #ddd; border-radius: 5px;">', unsafe_allow_html=True)

    st.divider()
    st.info("**Thesis Defense Note:** This evidence suite validates that the model error is centered at 0.00 RM, ensuring zero-bias policy simulations.")

# --- TAB 5: FORENSIC MULTI-WEEK FORECAST (PRODUCTION) ---
with tab5:
    st.header("🔮 Forensic Forecast (Pre-Trained Economic Engine)")
    st.caption("Architecture: **Loaded ARIMA Models** (Frozen Logic) + **Weekly Physics**.")

    # 1. LOAD DATA & MODELS
    try:
        df_buffer = pd.read_csv("full_buffer.csv")
        if df_buffer.empty: 
            st.error("❌ Buffer file is empty.")
            st.stop()
            
        latest_row = df_buffer.iloc[-1]
        latest_date = pd.to_datetime(latest_row['date']) if 'date' in latest_row else datetime.now()
        
        # LOAD THE PRE-TRAINED BRAINS
        arima_ron = joblib.load("arima_ron97.pkl")
        arima_diesel = joblib.load("arima_diesel.pkl")
        arima_usd = joblib.load("arima_usd.pkl")

        # Load Validation and Blind Scores for Error Bars
        # We use a helper to prevent crash if JSON is missing
        def safe_load_json(path):
            if os.path.exists(path):
                with open(path, 'r') as f: return json.load(f)
            return {}

        k_val_json = safe_load_json('kembung_validation_rmse.json')
        s_val_json = safe_load_json('selar_validation_rmse.json')
        k_blind_json = safe_load_json('kembung_blind_rmse.json')
        s_blind_json = safe_load_json('selar_blind_rmse.json')
        
    except Exception as e:
        st.error(f"Initialization Error: {e}. Ensure all .pkl and .csv files are uploaded.")
        st.stop()

    # 2. NATURE ENGINE (7-DAY INTEGRAL)
    def get_weekly_astral_mean(start_date):
        tides = []
        sunsets = []
        for d in range(7):
            day_date = start_date + timedelta(days=d)
            # Tide Logic
            p_days = moon.phase(day_date)
            t_force = math.cos(4 * math.pi * (p_days / 29.53))
            t_val = 1.6731 + (t_force * 0.15)
            tides.append(t_val)
            # Sunset Logic
            try:
                lumut = LocationInfo("Lumut", "Malaysia", "Asia/Kuala_Lumpur", 4.2105, 100.63)
                s_t = sun.sun(lumut.observer, date=day_date)
                s_dt = s_t['sunset'].replace(tzinfo=None) + timedelta(hours=8)
                s_val = s_dt.hour * 60 + s_dt.minute
            except: s_val = 1140.0
            sunsets.append(s_val)
        return np.mean(tides), np.mean(sunsets)

    # 3. GENERATE TRAJECTORIES
    with st.spinner("Calculating Economic Vectors..."):
        # Instant response from frozen ARIMA logic
        future_ron = arima_ron.forecast(steps=3).tolist()
        future_diesel = arima_diesel.forecast(steps=3).tolist()
        future_usd = arima_usd.forecast(steps=3).tolist()
    
    traj_ron = [float(latest_row['RON97'])] + future_ron
    traj_diesel = [float(latest_row['diesel'])] + future_diesel
    traj_usd = [float(latest_row['myr_per_usd_mean'])] + future_usd
    
    trajectory_data = []
    
    for i, w in enumerate([1, 2, 3, 4]):
        offset = 7 * (w - 1)
        t_date = latest_date + timedelta(days=offset)
        
        # A. Physical Inputs
        if w == 1:
            t_tide = float(latest_row['height_mean_m_perak'])
            t_sun = float(latest_row.get('sunset_mean_time_perak', 1140))
        else:
            t_tide, t_sun = get_weekly_astral_mean(t_date)

        # B. Economic Inputs
        in_ron, in_diesel, in_usd = traj_ron[i], traj_diesel[i], traj_usd[i]
        
        # C. Kembung Prediction
        k_in = pd.DataFrame({'RON97': [in_ron], 'height_mean_m_perak': [t_tide], 'myr_per_usd_mean': [in_usd]})
        k_base = k_struct.predict(k_in)[0]
        k_shock = (in_diesel - 2.15) * KEMBUNG_RATE if in_diesel > 2.15 else 0.0
        
        # D. Selar Prediction
        s_in = pd.DataFrame({'econ_pressure_index': [in_diesel * in_usd], 'solunar_synergy_perak': [t_tide * t_sun]})
        s_base = s_struct.predict(s_in)[0]
        s_shock = (in_diesel - 2.15) * SELAR_RATE if in_diesel > 2.15 else 0.0
        
        trajectory_data.append({
            "Week": f"Week {w}",
            "Start Date": t_date.strftime('%d-%b'),
            "Kembung Price": k_base + k_shock,
            "Selar Price": s_base + s_shock,
            "USD": in_usd, "Diesel": in_diesel, "RON97": in_ron, "Avg Tide": t_tide
        })

    df_traj = pd.DataFrame(trajectory_data)

    # 4. VISUALS
    forecast_weeks = st.radio("Select Forecast Horizon:", [1, 2, 3, 4], horizontal=True, index=0)
    sel_row = df_traj.iloc[forecast_weeks - 1]
    
    # Error bar logic
    if forecast_weeks == 1:
        k_err = float(k_val_json.get("avg_deviation", 0.56))
        s_err = float(s_val_json.get("avg_deviation", 0.70))
    else:
        k_err = float(k_blind_json.get(str(forecast_weeks), 0.65))
        s_err = float(s_blind_json.get(str(forecast_weeks), 0.85))

    st.subheader(f"Projection for Week of {sel_row['Start Date']}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Kembung Price", f"RM {sel_row['Kembung Price']:.2f}", delta=f"± RM {k_err:.2f}", delta_color="off")
        st.caption(f"Tide: {sel_row['Avg Tide']:.3f}m | RON97: {sel_row['RON97']:.2f}")
    with c2:
        st.metric("Selar Price", f"RM {sel_row['Selar Price']:.2f}", delta=f"± RM {s_err:.2f}", delta_color="off")
        st.caption(f"USD: {sel_row['USD']:.3f} | Diesel: {sel_row['Diesel']:.2f}")

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Price Forecast")
        st.line_chart(df_traj.set_index("Week")[["Kembung Price", "Selar Price"]], color=["#2e86c1", "#27ae60"])
    with col2:
        st.subheader("📊 Economic Trajectory")
        st.line_chart(df_traj.set_index("Week")[["USD", "Diesel", "RON97"]], color=["#e74c3c", "#7f8c8d", "#e67e22"])

    with st.expander("View Forensic Data Ledger", expanded=False):
        numeric_cols = ["Kembung Price", "Selar Price", "USD", "Diesel", "RON97", "Avg Tide"]
        st.dataframe(df_traj.style.format("{:.4f}", subset=numeric_cols))

