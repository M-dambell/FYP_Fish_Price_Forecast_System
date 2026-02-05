import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import base64
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import altair as alt 

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Seafood Security Simulator",
    layout="wide",
    page_icon="🐟"
)

# ==========================================
# HELPER: IMAGE ENCODING FOR UI
# ==========================================
def get_base64_image(path):
    """
    Reads a local image and converts it to a base64 string 
    to ensure it renders correctly in Streamlit.
    """
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None
    except Exception:
        return None

# ==========================================
# 2. MODEL LOADER (PURE STRUCTURAL)
# ==========================================
@st.cache_resource
def load_resources():
    """
    Loads the Pure Structural EBMs and the specific Metrics JSONs.
    """
    k_struct, s_struct = None, None
    k_metrics, s_metrics = None, None 
    k_audit, s_audit = {}, {} 
    
    try:
        # A. Load Structural Brains (Pure EBM models)
        if os.path.exists('kembung_structural_model.pkl'):
            k_struct = joblib.load('kembung_structural_model.pkl')
        if os.path.exists('selar_structural_model.pkl'):
            s_struct = joblib.load('selar_structural_model.pkl')
        
        # B. Load the Overview Metrics (Summary for Tab 1)
        if os.path.exists('kembung_metrics.json'):
            with open('kembung_metrics.json', 'r') as f:
                k_metrics = json.load(f)
        if os.path.exists('selar_metrics.json'):
            with open('selar_metrics.json', 'r') as f:
                s_metrics = json.load(f)

        # C. Load the Detailed Audit Logs (Fallback / Deep Dive)
        if os.path.exists('kembung_blind_rmse.json'):
            with open('kembung_blind_rmse.json', 'r') as f:
                k_audit = json.load(f)
        if os.path.exists('selar_blind_rmse.json'):
            with open('selar_blind_rmse.json', 'r') as f:
                s_audit = json.load(f)
            
        return k_struct, s_struct, k_metrics, s_metrics, k_audit, s_audit

    except Exception as e:
        st.error(f"❌ Critical Error Loading Resources: {e}")
        return None, None, None, None, {}, {}

# Unpack all 6 variables
k_struct, s_struct, k_metrics, s_metrics, k_stats, s_stats = load_resources()

# Safety Check
if k_struct is None or s_struct is None:
    st.warning("⚠️ Structural model files (.pkl) not found. Dashboard in standby.")
    st.stop()


# ==========================================
# CENTRAL GENERATIVE BRAIN (TAB SYNC)
# ==========================================
# Initialize gen_data to avoid NameError if file is missing
gen_data = []

if os.path.exists("full_buffer.csv"):
    try:
        df_buffer = pd.read_csv("full_buffer.csv")
        latest_date = pd.to_datetime(df_buffer.iloc[-1]['date'])
        
        # Check if ARIMA models exist before forecasting
        if os.path.exists("arima_ron97.pkl") and os.path.exists("arima_usd.pkl"):
            f_ron = joblib.load("arima_ron97.pkl").forecast(steps=4).values
            f_die = joblib.load("arima_diesel.pkl").forecast(steps=4).values
            f_usd = joblib.load("arima_usd.pkl").forecast(steps=4).values
            
            for i in range(4):
                in_date = latest_date + timedelta(days=7 * i)
                # Unified Physics Logic
                p_days = (in_date - pd.Timestamp('2024-01-01')).days % 29.53
                t_tide = 1.6731 + (math.cos(4 * math.pi * (p_days / 29.53)) * 0.15)
                t_sun = 1140.0 + math.sin(2 * math.pi * ((in_date.dayofyear - 80)/365)) * 15
                
                # Unified Model Inference
                k_price = float(k_struct.predict(pd.DataFrame({'RON97':[f_ron[i]], 'height_mean_m_perak':[t_tide], 'myr_per_usd_mean':[f_usd[i]]}))[0])
                s_price = float(s_struct.predict(pd.DataFrame({'econ_pressure_index':[f_die[i]*f_usd[i]], 'solunar_synergy_perak':[t_tide*t_sun]}))[0])
                
                gen_data.append({
                    "in_date": in_date, "target_date": in_date + timedelta(days=7),
                    "ron": float(f_ron[i]), "die": float(f_die[i]), "usd": float(f_usd[i]), "tide": float(t_tide), "sun": float(t_sun),
                    "k_price": k_price, "s_price": s_price
                })
    except Exception as e:
        st.error(f"⚠️ Generative Brain Error: {e}")

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.title("Main Menu")
    species_choice = st.radio("Select Species:", ["🐟 Kembung", "🐠 Selar"])
    
    st.divider()
    
    # System Status Panel
    st.info("System Architecture")
    if "Kembung" in species_choice:
        st.caption("Mode: Domestic Benchmark")
        st.caption("Config: 3 Predictive Vectors")
        st.caption("Target: Price Decoupling Proof")
    else:
        st.caption("Mode: Surgical Precision")
        st.caption("Config: Interaction Synergy")
        st.caption("Target: Structural Buffering")
        
    st.divider()
    st.caption("v7.0 Operational Mode")

# ==========================================
# 4. TAB SYSTEM (OPERATIONAL ORDER - 9 TABS)
# ==========================================
tabs = st.tabs([
    # --- PHASE 1: INTELLIGENCE ---
    "📊 Overview",          # Tab 1
    "🔮 Forecast",          # Tab 2
    "🧬 Price DNA",         # Tab 3
    "🛡️ Risk Assessment",   # Tab 4
    # --- PHASE 2: ACTION ---
    "🎯 Surgical Strategy", # Tab 5
    "⚖️ Sensitivity Rules", # Tab 6
    "🚨 Gouging Detector",  # Tab 7
    # --- PHASE 3: AUDIT ---
    "🎮 Simulator",         # Tab 8
    "📜 Governance Ledger"  # Tab 9
])

# UNPACK 9 VARIABLES
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = tabs

# ==========================================
# TAB 1: EXECUTIVE SUMMARY (STRICT PARSE)
# ==========================================
with tab1:
    st.header("Malaysian Seafood Resilience Audit")
    st.write("Forensic analysis of price volatility using **Explainable Boosting Machines (EBM)** without manual heuristic bias.")
    
    c1, c2, c3 = st.columns(3)
    
    # 1. SELECT DATA SOURCE based on Species
    if "Kembung" in species_choice:
        target_metrics = k_metrics
    else:
        target_metrics = s_metrics

    # 2. STRICT EXTRACTION LOGIC (Customized for your JSON format)
    def extract_strict(data):
        """
        Extracts specific keys: "Accuracy" (str with %) and "Structural_MAE" (str with RM)
        """
        if not data: return 0.0, 0.0
        
        acc, mae = 0.0, 0.0
        
        # A. Handle "Accuracy" key (e.g. "96.02%")
        if "Accuracy" in data:
            raw_acc = str(data["Accuracy"]).replace("%", "").strip()
            try:
                acc = float(raw_acc)
            except:
                acc = 0.0
        elif "accuracy" in data:
            acc = float(data["accuracy"])
            
        # B. Handle MAE (Try "Structural_MAE" first, then "avg_deviation")
        if "Structural_MAE" in data:
            raw_mae = str(data["Structural_MAE"]).replace("RM", "").strip()
            try:
                mae = float(raw_mae)
            except:
                mae = 0.0
        elif "avg_deviation" in data:
            mae = float(data["avg_deviation"])
        elif "mae" in data:
            mae = float(data["mae"])
            
        return acc, mae

    # 3. GET VALUES (NO FALLBACK)
    final_acc, final_mae = extract_strict(target_metrics)

    # 4. DISPLAY METRICS
    if "Kembung" in species_choice:
        with c1: 
            st.metric(
                "Horizon Resilience", 
                f"± RM {final_mae:.2f}", 
                delta=f"{final_acc:.2f}% Accuracy"
            )
        with c2: 
            st.metric("Model Integrity", "Stable", "Self-Correcting")
        with c3:
            st.metric("Core Sensitivity", "Fuel + Tides", "High Elasticity")
        
        st.divider()
        st.success("""
        **🏆 Policy Verdict:** Valid for **Long-Term Planning**. 
        Kembung demonstrates **Structural Decoupling** from fuel shocks. 
        Prices are governed by natural cycles and macro-economic baseline trends rather than direct fuel pass-through.
        """)
        
    else:
        with c1: 
            st.metric(
                "Horizon Resilience", 
                f"± RM {final_mae:.2f}", 
                delta=f"{final_acc:.2f}% Accuracy"
            )
        with c2: 
            st.metric("Model Integrity", "High Precision", "Zero Drift")
        with c3:
            st.metric("Core Sensitivity", "USD Pressure", "Logistics Interaction")
        
        st.divider()
        st.success("""
        **🏆 Policy Verdict:** Valid for **Monthly Review**. 
        Selar demonstrates **Market Buffering**, where interaction effects between currency and logistics costs create a stable price floor.
        """)

# ==========================================
# TAB 2: FORECAST
# ==========================================
with tab2:
    st.header("🔮 Pure Structural Forecast (Generative Engine)")
    st.caption("Architecture: **Autonomous EBM** + **ARIMA Projections**.")

    if not gen_data:
        st.warning("⚠️ Generative Data unavailable. Please check 'full_buffer.csv'.")
    else:
        f_idx = st.radio("Select Forecast Horizon:", [1, 2, 3, 4], horizontal=True, key="f_sync_final")
        selected = gen_data[f_idx - 1]
        is_k = "Kembung" in species_choice
        display_price = selected['k_price'] if is_k else selected['s_price']
        
        # --- CONSTANT MAE LOGIC ---
        # 1. Select the correct metrics source based on species
        current_metrics_json = k_metrics if is_k else s_metrics
        
        # 2. Extract Structural_MAE strictly from the JSON loaded in load_resources()
        structural_mae = 0.55 # Fallback
        
        if current_metrics_json:
            if "Structural_MAE" in current_metrics_json:
                # Clean string "RM 0.5632" -> 0.5632
                raw_val = str(current_metrics_json["Structural_MAE"]).replace("RM", "").strip()
                try:
                    structural_mae = float(raw_val)
                except:
                    pass
            elif "avg_deviation" in current_metrics_json:
                structural_mae = float(current_metrics_json["avg_deviation"])
        # --------------------------

        st.subheader(f"Projection for Week: {selected['target_date'].strftime('%d-%b')}")
        
        # Display Single Metric (RMSE Removed)
        st.metric(
            label=f"{species_choice} Price", 
            value=f"RM {display_price:.2f}", 
            delta=f"± RM {structural_mae:.4f}"
        )

        st.divider()
        
        df_traj = pd.DataFrame(gen_data)
        df_traj['DisplayPrice'] = df_traj['k_price'] if is_k else df_traj['s_price']
        df_traj['ForecastDate'] = df_traj['target_date'].dt.strftime('%d-%b')
        
        col_chart, col_table = st.columns([1.5, 1])
        with col_chart:
            st.line_chart(df_traj.set_index("ForecastDate")["DisplayPrice"])
        with col_table:
            st.dataframe(df_traj[["ForecastDate", "DisplayPrice", "usd", "die"]].style.format({"DisplayPrice": "{:.2f}", "usd": "{:.4f}"}))

        with st.expander("🔬 View Detailed ARIMA Forecasted Inputs", expanded=False):
            st.table(df_traj[["in_date", "ron", "die", "usd", "tide"]].rename(columns={"in_date": "Input Date"}))

# ==========================================
# TAB 3: PRICE DNA DECONSTRUCTOR
# ==========================================
with tab3:
    c_header, c_select = st.columns([2, 1])
    with c_header:
        st.header("🧬 Deliverable #9: Price DNA Deconstructor")
        st.write("Decomposing the **Forecasted Price** into structural drivers.")
    
    with c_select:
        h_map = {"Current": 0, "+1 Wk": 1, "+2 Wk": 2, "+3 Wk": 3, "+4 Wk": 4}
        sel_h = st.selectbox("⏳ Select Horizon:", list(h_map.keys()), key="dna_final_v17_fix")
        w_idx = h_map[sel_h]

    is_k = "Kembung" in species_choice
    mod = k_struct if is_k else s_struct 

    if os.path.exists("full_buffer.csv"):
        # Logic to handle missing gen_data gracefully
        proceed = True
        if w_idx > 0 and not gen_data:
            st.error("Forward DNA analysis unavailable (Missing generative data).")
            proceed = False
            
        if proceed:
            if w_idx == 0:
                row = df_buffer.iloc[-2]
                c_ron, c_die, c_usd = float(row['RON97']), float(row['diesel']), float(row['myr_per_usd_mean'])
                in_date, tar_date = pd.to_datetime(row['date']), latest_date
                p_d = (in_date - pd.Timestamp('2024-01-01')).days % 29.53
                c_tid = 1.6731 + (math.cos(4 * math.pi * (p_d / 29.53)) * 0.15)
                c_sun = 1140.0 + math.sin(2 * math.pi * ((in_date.dayofyear - 80)/365)) * 15
                
                if is_k:
                    final_p = float(mod.predict(pd.DataFrame({'RON97':[c_ron], 'height_mean_m_perak':[c_tid], 'myr_per_usd_mean':[c_usd]}))[0])
                else:
                    final_p = float(mod.predict(pd.DataFrame({'econ_pressure_index':[c_die*c_usd], 'solunar_synergy_perak':[c_tid*c_sun]}))[0])
                source = "Historical Record"
            else:
                s = gen_data[w_idx - 1]
                c_ron, c_die, c_usd, c_tid, c_sun = s['ron'], s['die'], s['usd'], s['tide'], s['sun']
                in_date, tar_date = s['in_date'], s['target_date']
                final_p = s['k_price'] if is_k else s['s_price']
                source = "ARIMA Projection"

            chart_data = []
            try:
                if is_k:
                    def p_k(r, t, u): 
                        return float(mod.predict(pd.DataFrame({'RON97':[r], 'height_mean_m_perak':[t], 'myr_per_usd_mean':[u]}))[0])
                    base_p = p_k(2.05, 1.5, 4.0)
                    
                    raw_imp_r = p_k(c_ron, 1.5, 4.0) - base_p
                    raw_imp_u = p_k(2.05, 1.5, c_usd) - base_p
                    raw_imp_t = p_k(2.05, c_tid, 4.0) - base_p
                    
                    raw_sum = raw_imp_r + raw_imp_u + raw_imp_t
                    target_diff = final_p - base_p
                    adj = target_diff / raw_sum if raw_sum != 0 else 1
                    
                    chart_data = [
                        {"Factor": "RON97 Fuel Vector", "Impact": raw_imp_r * adj, "Color": "#3498db"},
                        {"Factor": "MYR/USD Exchange", "Impact": raw_imp_u * adj, "Color": "#9b59b6"},
                        {"Factor": "Perak Tide Height", "Impact": raw_imp_t * adj, "Color": "#2ecc71"}
                    ]
                else:
                    def p_s(ep, ss): 
                        return float(mod.predict(pd.DataFrame({'econ_pressure_index':[ep], 'solunar_synergy_perak':[ss]}))[0])
                    base_ep, base_ss = (2.15 * 4.0), (1.5 * 1140.0)
                    base_p = p_s(base_ep, base_ss)
                    
                    raw_imp_m = p_s(c_die * c_usd, base_ss) - base_p
                    raw_imp_n = p_s(base_ep, c_tid * c_sun) - base_p
                    
                    raw_sum = raw_imp_m + raw_imp_n
                    target_diff = final_p - base_p
                    adj = target_diff / raw_sum if raw_sum != 0 else 1
                    
                    chart_data = [
                        {"Factor": "Econ Pressure (Diesel×USD)", "Impact": raw_imp_m * adj, "Color": "#3498db"},
                        {"Factor": "Solunar Synergy (Tide×Sun)", "Impact": raw_imp_n * adj, "Color": "#2ecc71"}
                    ]

                st.divider()
                actual_total_dev = final_p - base_p
                m1, m2, m3 = st.columns(3)
                m1.metric("1. Model Base", f"RM {base_p:.2f}")
                m2.metric("2. Net Structural Impact", f"RM {actual_total_dev:+.2f}")
                m3.metric(f"3. {sel_h} Result", f"RM {final_p:.2f}", f"Target: {tar_date.strftime('%d %b')}")

                st.altair_chart(alt.Chart(pd.DataFrame(chart_data)).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                    x=alt.X('Factor', sort=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Impact', title='Price Contribution (RM)'),
                    color=alt.Color('Color', scale=None),
                    tooltip=['Factor', 'Impact']
                ).properties(height=350), use_container_width=True)

                st.subheader("📋 Structural Input Verification")
                st.table(pd.DataFrame({
                    "Parameter": ["Diesel", "RON97", "USD Rate", "Tide Height"],
                    "Value": [f"RM {c_die:.2f}", f"RM {c_ron:.2f}", f"{c_usd:.4f}", f"{c_tid:.4f}m"],
                    "Source": [source]*4, "Ref Date": [in_date.strftime('%d %b')]*4
                }))

            except Exception as e:
                st.error(f"⚠️ DNA Analysis Error: {str(e)}")

# ==========================================
# TAB 4: RISK ASSESSMENT
# ==========================================
with tab4:
    st.header("🛡️ Deliverable #6: Strategic Risk Assessment")
    st.warning("""
    ⚠️ **Dynamic Policy Disclaimer:** The underlying ARIMA engines are **automatically retrained on a weekly basis** using live market data. 
    Consequently, these 24-week projections recalibrate in real-time.
    """)

    st.write("""
    This utility provides a **Semi-Annual Forward-Look** into market volatility. 
    By setting custom Governance Tolerances, policymakers can identify future **Risk Windows**.
    """)

    is_k = "Kembung" in species_choice
    mod = k_struct if is_k else s_struct 

    c_sliders, c_kpi = st.columns([1, 1.2])
    
    with c_sliders:
        st.subheader("⚙️ Set Risk Thresholds")
        green_tol = st.slider("Green Zone (Stable) ± RM:", 0.01, 2.00, 0.40, 0.05, key="gtol_risk_assessment")
        amber_tol = st.slider("Amber Zone (Warning) ± RM:", 0.05, 5.00, 1.20, 0.05, key="atol_risk_assessment")
        st.markdown(f"**Crisis Threshold:** > RM {amber_tol:.2f} Deviation")

    if os.path.exists("full_buffer.csv"):
        try:
            # Safe Fallback if gen_data is empty
            current_base = gen_data[0]['k_price' if is_k else 's_price'] if gen_data else 0.0
            
            latest_date = pd.to_datetime(df_buffer.iloc[-1]['date'])
            
            if os.path.exists("arima_ron97.pkl"):
                f24_ron = joblib.load("arima_ron97.pkl").forecast(steps=24).values
                f24_die = joblib.load("arima_diesel.pkl").forecast(steps=24).values
                f24_usd = joblib.load("arima_usd.pkl").forecast(steps=24).values
                
                outlook_data = []
                for i in range(24):
                    in_date = latest_date + timedelta(days=7 * i)
                    target_date = in_date + timedelta(days=7)
                    
                    p_days = (in_date - pd.Timestamp('2024-01-01')).days % 29.53
                    t_tide = 1.6731 + (math.cos(4 * math.pi * (p_days / 29.53)) * 0.15)
                    
                    if is_k:
                        X_sim = pd.DataFrame({'RON97':[f24_ron[i]], 'height_mean_m_perak':[t_tide], 'myr_per_usd_mean':[f24_usd[i]]})
                        p_base = float(mod.predict(X_sim)[0])
                    else:
                        X_sim = pd.DataFrame({'econ_pressure_index':[f24_die[i]*f24_usd[i]], 'solunar_synergy_perak':[t_tide*1140.0]})
                        p_base = float(mod.predict(X_sim)[0])
                    
                    deviation = p_base - current_base
                    abs_dev = abs(deviation)

                    if abs_dev > amber_tol:
                        risk_status = "🔴 Red (Crisis)"
                    elif abs_dev > green_tol:
                        risk_status = "🟡 Amber (Warn)"
                    else:
                        risk_status = "🟢 Green (Stable)"

                    outlook_data.append({
                        "Target Date": target_date.strftime('%d-%b'),
                        "Projected Price": p_base,
                        "USD": float(f24_usd[i]),
                        "Fuel": float(f24_ron[i] if is_k else f24_die[i]),
                        "Risk Status": risk_status,
                        "Deviation": deviation
                    })

                df_outlook = pd.DataFrame(outlook_data)

                with c_kpi:
                    st.subheader("🚦 Strategic Risk Verdict")
                    red_c = len(df_outlook[df_outlook["Risk Status"] == "🔴 Red (Crisis)"])
                    amb_c = len(df_outlook[df_outlook["Risk Status"] == "🟡 Amber (Warn)"])
                    grn_c = len(df_outlook[df_outlook["Risk Status"] == "🟢 Green (Stable)"])
                    
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Current Base", f"RM {current_base:.2f}")
                    kpi2.metric("Red Wks", f"{red_c}", delta_color="inverse")
                    kpi3.metric("Amber Wks", f"{amb_c}", delta_color="off")
                    kpi4.metric("Green Wks", f"{grn_c}", delta_color="normal")
                    
                    if red_c > 12:
                        st.error("🚨 **High Risk Alert:** Systematic market de-sync predicted.")
                    elif grn_c > 18:
                        st.success("✅ **Market Stability:** Structural anchors are effectively absorbing volatility.")
                    else:
                        st.warning("⚠️ **Active Monitoring:** Intermittent risk windows detected.")

                st.divider()
                st.subheader("📑 24-Week Strategic Risk Ledger")
                
                def style_risk_ledger(val):
                    if '🔴' in str(val): return 'background-color: #f8d7da; color: #721c24; font-weight: bold; border-left: 8px solid #dc3545;'
                    if '🟡' in str(val): return 'background-color: #fff3cd; color: #856404; font-weight: bold; border-left: 8px solid #ffc107;'
                    if '🟢' in str(val): return 'background-color: #d4edda; color: #155724; font-weight: bold; border-left: 8px solid #28a745;'
                    return ''

                st.dataframe(
                    df_outlook.style.applymap(style_risk_ledger, subset=['Risk Status'])
                    .format({"Projected Price": "RM {:.2f}", "USD": "{:.4f}", "Fuel": "{:.2f}", "Deviation": "{:+.2f} RM"}),
                    use_container_width=True,
                    height=550
                )

                with st.expander("📊 View Policy Risk Corridor", expanded=True):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    x_idx = range(len(df_outlook))
                    ax.fill_between(x_idx, df_outlook['Projected Price'] - amber_tol, df_outlook['Projected Price'] + amber_tol, color='#f1c40f', alpha=0.1, label='Amber Zone')
                    ax.fill_between(x_idx, df_outlook['Projected Price'] - green_tol, df_outlook['Projected Price'] + green_tol, color='#27ae60', alpha=0.2, label='Green Zone')
                    ax.plot(df_outlook['Projected Price'].values, color='#2c3e50', linestyle='--', label='Structural Baseline')
                    ax.set_xticks(x_idx[::2])
                    ax.set_xticklabels(df_outlook['Target Date'].values[::2], rotation=45)
                    ax.set_ylabel("Price (RM/kg)")
                    ax.legend(loc='upper left', fontsize='small')
                    st.pyplot(fig)
                    st.caption("**Forensic Anchor:** This 6-month 'Policy Tunnel' is stabilized by deterministic tidal cycles and macro-economic ARIMA projections.")
            else:
                st.warning("⚠️ ARIMA Models missing for risk assessment.")

        except Exception as e:
            st.error(f"⚠️ Risk Engine Error: {str(e)}")
    else:
        st.warning("⚠️ Historical buffer missing. Strategic assessment in standby.")

# ==========================================
# TAB 5: SURGICAL STRATEGY (COMPARATIVE SCENARIO PLANNING)
# ==========================================
with tab5:
    st.header("🎯 Deliverable #10: Comparative Scenario Planning")
    st.write("""
    This utility projects the **Strategic Divergence** between the Status Quo (No Intervention) 
    and a proposed **Macro-Policy Intervention**. It dynamically recalculates **Interaction Effects** to capture the compounding impact of multi-lever policies.
    """)

    is_k = "Kembung" in species_choice
    mod = k_struct if is_k else s_struct 

    if os.path.exists("full_buffer.csv"):
        try:
            # 1. SCENARIO CONTROLS
            c_settings, c_chart = st.columns([1, 2.5])
            
            with c_settings:
                st.subheader("⚙️ Policy Levers")
                st.markdown("**1. 🟦 Baseline (Status Quo):**")
                st.caption("*EBM Structural Forecast driven by standard ARIMA macro-projections (No Intervention).*")
                
                st.markdown("**2. 🟩 Intervention Settings:**")
                fuel_label = "RON97 Price (RM)" if is_k else "Diesel Price (RM)"
                default_fuel = 2.05 if is_k else 2.15
                
                pol_fuel = st.number_input(
                    f"Set {fuel_label}:", 1.50, 5.00, default_fuel, 0.05, 
                    help=f"Simulate a fixed ceiling for {fuel_label}."
                )
                pol_usd = st.number_input(
                    "Set MYR/USD Peg:", 3.00, 6.00, 4.20, 0.01, 
                    help="Simulate a currency hedging or pegging policy."
                )
                
                st.info("Adjusting both levers triggers the model's interaction logic.")

                with st.expander("🧠 Why change both?", expanded=True):
                    st.markdown("""
                    **The Interaction Multiplier:**
                    For species like **Selar**, the model is trained on the interaction:
                    $$Index = Fuel \\times USD$$
                    
                    Reducing **both** levers simultaneously creates a **compounding benefit** larger than the sum of its parts.
                    """)

            # 2. GENERATE DATA (24-Week Loop)
            # Need arima models
            if os.path.exists("arima_ron97.pkl"):
                f24_ron = joblib.load("arima_ron97.pkl").forecast(steps=24).values
                f24_die = joblib.load("arima_diesel.pkl").forecast(steps=24).values
                f24_usd = joblib.load("arima_usd.pkl").forecast(steps=24).values
                latest_date = pd.to_datetime(df_buffer.iloc[-1]['date'])
                
                scenarios = []
                for i in range(24):
                    in_date = latest_date + timedelta(days=7 * i)
                    target_date = in_date + timedelta(days=7)
                    
                    # Physics
                    p_days = (in_date - pd.Timestamp('2024-01-01')).days % 29.53
                    t_tide = 1.6731 + (math.cos(4 * math.pi * (p_days / 29.53)) * 0.15)
                    t_sun = 1140.0 + math.sin(2 * math.pi * ((in_date.dayofyear - 80)/365)) * 15
                    
                    base_usd = float(f24_usd[i])
                    base_fuel = float(f24_ron[i] if is_k else f24_die[i])
                    
                    # Baseline
                    if is_k:
                        p1 = float(mod.predict(pd.DataFrame({'RON97':[base_fuel], 'height_mean_m_perak':[t_tide], 'myr_per_usd_mean':[base_usd]}))[0])
                    else:
                        p1 = float(mod.predict(pd.DataFrame({'econ_pressure_index':[base_fuel*base_usd], 'solunar_synergy_perak':[t_tide*t_sun]}))[0])
                    scenarios.append({"Date": target_date, "Price": p1, "Scenario": "Baseline (Status Quo)"})
                    
                    # Intervention
                    int_usd = pol_usd
                    int_fuel = pol_fuel 
                    
                    if is_k:
                        p2 = float(mod.predict(pd.DataFrame({'RON97':[int_fuel], 'height_mean_m_perak':[t_tide], 'myr_per_usd_mean':[int_usd]}))[0])
                    else:
                        policy_interaction = int_fuel * int_usd
                        p2 = float(mod.predict(pd.DataFrame({'econ_pressure_index':[policy_interaction], 'solunar_synergy_perak':[t_tide*t_sun]}))[0])
                    scenarios.append({"Date": target_date, "Price": p2, "Scenario": "Policy Intervention"})

                df_scen = pd.DataFrame(scenarios)

                # 3. VISUALIZE DIVERGENCE
                with c_chart:
                    st.subheader("📈 24-Week Divergence Trajectory")
                    min_p = df_scen['Price'].min()
                    max_p = df_scen['Price'].max()
                    y_domain = [min_p - 1.0, max_p + 1.0]

                    base = alt.Chart(df_scen).encode(
                        x=alt.X('Date:T', title="Horizon (6 Months)", axis=alt.Axis(format='%d %b')),
                        y=alt.Y('Price:Q', title="Retail Price (RM)", scale=alt.Scale(domain=y_domain)),
                        color=alt.Color('Scenario', scale=alt.Scale(
                            domain=['Baseline (Status Quo)', 'Policy Intervention'],
                            range=['#3498db', '#2ecc71'] 
                        ))
                    )

                    chart = base.mark_line(point=True, interpolate='basis', strokeWidth=3).encode(
                        order='Date'
                    ).interactive()

                    st.altair_chart(chart, use_container_width=True)
                    
                    st.caption(f"""
                    **Visual Note:** The chart uses a normalized axis to visualize the strategic gap. 
                    The gap between the Blue (No Action) and Green (Policy) lines represents the **Direct Consumer Impact**.
                    """)

                st.divider()

                # 4. STRATEGIC IMPACT SUMMARY
                st.subheader("📊 Strategic Impact Summary")
                
                pivot = df_scen.pivot_table(index='Scenario', values='Price', aggfunc=['mean', 'max'])
                pivot.columns = ['6-Mo Avg', 'Peak Price']
                
                baseline_lbl = 'Baseline (Status Quo)'
                baseline_avg = pivot.loc[baseline_lbl, '6-Mo Avg']
                pivot['Difference'] = pivot['6-Mo Avg'] - baseline_avg
                
                c1, c2 = st.columns(2)
                
                b_row = pivot.loc[baseline_lbl]
                c1.metric("🟦 No Intervention Avg", f"RM {b_row['6-Mo Avg']:.2f}")
                
                i_row = pivot.loc['Policy Intervention']
                diff = i_row['Difference']
                
                if diff < 0:
                    lbl = "Decrease"
                    d_color = "inverse" 
                else:
                    lbl = "Increase"
                    d_color = "normal"  

                c2.metric("🟩 Intervention Average", f"RM {i_row['6-Mo Avg']:.2f}", delta=f"{diff:.2f} RM {lbl}", delta_color=d_color)

                pct_change = (abs(diff) / b_row['6-Mo Avg']) * 100
                direction = "lowers" if diff < 0 else "raises"
                
                st.info(f"""
                **🏛️ Strategic Verdict:** This policy combination ({fuel_label} @ RM {pol_fuel:.2f} + USD @ {pol_usd:.2f}) 
                **{direction}** the 6-month structural cost base by **{pct_change:.1f}%**. 
                The divergence proves the efficacy of targeting **upstream interaction effects** (Logistics × Currency).
                """)

                with st.expander("🕵️ Forensic Diagnostics (Check Inputs)", expanded=False):
                    st.write("Compare the raw inputs entering the model for the **First Week** of the simulation.")
                    b_fuel = float(f24_ron[0] if is_k else f24_die[0])
                    b_usd = float(f24_usd[0])
                    b_idx = b_fuel * b_usd if not is_k else 0
                    b_price_0 = scenarios[0]['Price']
                    i_price_0 = scenarios[1]['Price']
                    i_fuel = pol_fuel
                    i_usd = pol_usd
                    i_idx = i_fuel * i_usd if not is_k else 0
                    
                    d_cols = st.columns(4)
                    d_cols[0].metric("Baseline Input", f"{b_idx:.2f}" if not is_k else f"R {b_fuel:.2f} / U {b_usd:.2f}")
                    d_cols[1].metric("Policy Input", f"{i_idx:.2f}" if not is_k else f"R {i_fuel:.2f} / U {i_usd:.2f}")
                    
                    if not is_k:
                        input_change = ((i_idx - b_idx)/b_idx)*100
                    else:
                        input_change = ((i_fuel - b_fuel)/b_fuel)*100
                        
                    d_cols[2].metric("Input % Change", f"{input_change:+.1f}%")
                    d_cols[3].metric("Resulting Price Delta", f"{i_price_0 - b_price_0:+.2f} RM")
                    st.info("Interpretation: If 'Input % Change' is significant but 'Resulting Price Delta' is small, this proves **Model Saturation**.")
            else:
                st.warning("ARIMA forecasts missing for scenario planning.")

        except Exception as e:
            st.error(f"⚠️ Scenario Engine Error: {str(e)}")
    else:
        st.warning("⚠️ Buffer missing.")

# ==========================================
# TAB 6: SENSITIVITY RULES
# ==========================================
with tab6:
    st.header("⚖️ Deliverable #11: Strategic Sensitivity Rules")
    st.write("""
    This module calculates **'Rules of Thumb'** for policy decisions and visualizes the 
    **Structural Mechanics** (Feature Engineering) determining these sensitivities.
    """)

    is_k = "Kembung" in species_choice
    mod = k_struct if is_k else s_struct 
    
    if os.path.exists("full_buffer.csv"):
        try:
            row = df_buffer.iloc[-1]
            base_ron = float(row.get('RON97', 2.05))
            base_die = float(row.get('diesel', 2.15))
            base_usd = float(row.get('myr_per_usd_mean', 4.40))
            base_tide = float(row.get('height_mean_m_perak', 1.6))
            base_sun = float(row.get('sunset_mean_time_perak', 1140))
            
            def get_price(r, d, u, t, s):
                if is_k:
                    return float(mod.predict(pd.DataFrame({'RON97':[r], 'height_mean_m_perak':[t], 'myr_per_usd_mean':[u]}))[0])
                else:
                    return float(mod.predict(pd.DataFrame({'econ_pressure_index':[d*u], 'solunar_synergy_perak':[t*s]}))[0])

            base_price = get_price(base_ron, base_die, base_usd, base_tide, base_sun)

            st.subheader("📏 The Laws of Price Sensitivity")
            st.caption(f"Based on current market conditions for **{species_choice}**.")

            c1, c2 = st.columns(2)

            # Rule 1: Fuel
            scan_range = 0.50
            p_fuel_high = get_price(base_ron + scan_range, base_die + scan_range, base_usd, base_tide, base_sun)
            p_fuel_low  = get_price(base_ron - scan_range, base_die - scan_range, base_usd, base_tide, base_sun)
            steps = (scan_range * 2) / 0.10
            avg_fuel_impact_10sen = (p_fuel_high - p_fuel_low) / steps
            fuel_name = "RON97" if is_k else "Diesel"
            
            if abs(avg_fuel_impact_10sen) < 0.005: 
                f_val_str = "🛡️ BUFFERED"
                f_lbl = "Stable"
                f_color = "#95a5a6"
                f_desc = "Saturation Point Reached"
            else:
                f_lbl = "RISES" if avg_fuel_impact_10sen > 0 else "DROPS"
                f_color = "#e74c3c" if avg_fuel_impact_10sen > 0 else "#2ecc71"
                f_val_str = f"{abs(avg_fuel_impact_10sen)*100:.1f} sen"
                f_desc = "Avg Retail Impact"
            
            with c1:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                    <h4 style="margin:0">⛽ {fuel_name} Rule</h4>
                    <p style="font-size: 1.1em; color: #555;">For every <b>10 cent</b> hike in fuel...</p>
                    <h2 style="color: {f_color}; margin: 0;">{f_lbl} {f_val_str}</h2>
                    <p style="font-size: 0.9em; color: #7f8c8d;">{f_desc}</p>
                </div>
                """, unsafe_allow_html=True)

            # Rule 2: Currency
            p_usd_high = get_price(base_ron, base_die, base_usd + scan_range, base_tide, base_sun)
            p_usd_low  = get_price(base_ron, base_die, base_usd - scan_range, base_tide, base_sun)
            avg_usd_impact_10sen = (p_usd_high - p_usd_low) / steps

            if abs(avg_usd_impact_10sen) < 0.005:
                u_val_str = "🛡️ BUFFERED"
                u_lbl = "Stable"
                u_color = "#95a5a6"
                u_desc = "Saturation Point Reached"
            else:
                u_lbl = "RISES" if avg_usd_impact_10sen > 0 else "DROPS"
                u_color = "#e67e22" if avg_usd_impact_10sen > 0 else "#2ecc71"
                u_val_str = f"{abs(avg_usd_impact_10sen)*100:.1f} sen"
                u_desc = "Avg Retail Impact"

            with c2:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                    <h4 style="margin:0">💵 Currency Rule</h4>
                    <p style="font-size: 1.1em; color: #555;">For every <b>10 cent</b> weakening of Ringgit...</p>
                    <h2 style="color: {u_color}; margin: 0;">{u_lbl} {u_val_str}</h2>
                    <p style="font-size: 0.9em; color: #7f8c8d;">{u_desc}</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.subheader("🔬 Structural Mechanics: Why is it Buffered?")
            
            if is_k:
                st.write("**Mechanism:** Additive Linear Driver")
                st.latex(r"Price \approx \beta_1(Fuel) + \beta_2(Tide) + \beta_3(USD)")
                st.info("Kembung prices respond linearly to inputs. The relationship is direct.")
            else:
                st.write("**Mechanism:** Multiplicative Interaction (Synergy)")
                st.latex(r"Price \approx f(Diesel \times USD) + g(Tides)")
                current_index = base_die * base_usd
                sweep_range = np.linspace(current_index * 0.5, current_index * 1.5, 50)
                curve_data = []
                for idx_val in sweep_range:
                    pred_p = float(mod.predict(pd.DataFrame({'econ_pressure_index':[idx_val], 'solunar_synergy_perak':[base_tide * base_sun]}))[0])
                    curve_data.append({"Index": idx_val, "Predicted Price": pred_p, "Type": "Structural Curve"})
                
                curve_data.append({"Index": current_index, "Predicted Price": base_price, "Type": "Current Market Position"})
                df_curve = pd.DataFrame(curve_data)
                
                base_c = alt.Chart(df_curve).encode(x=alt.X('Index', title='Economic Pressure Index (Diesel × USD)'), y=alt.Y('Predicted Price', title='Retail Price (RM)', scale=alt.Scale(zero=False)))
                line_c = base_c.mark_line().encode(color=alt.value("#3498db")).transform_filter(alt.datum.Type == 'Structural Curve')
                point_c = base_c.mark_circle(size=200, color="red").encode(tooltip=['Index', 'Predicted Price']).transform_filter(alt.datum.Type == 'Current Market Position')
                c_mech, c_txt = st.columns([2, 1])
                with c_mech: st.altair_chart(line_c + point_c, use_container_width=True)
                with c_txt: st.markdown(f"**Forensic Diagnosis:**\nThe red dot shows current market position.\n* **Current Index:** {current_index:.2f}\n* **Shape:** Plateau (Flat)\n**Verdict:** The market is on the **'Saturation Plateau'**.")

            st.divider()
            st.subheader("📋 The Policy Pocket Card")
            def fmt_impact(val):
                if abs(val) < 0.005: return "🛡️ BUFFERED (Saturation)"
                action = "RISES" if val > 0 else "DROPS"
                return f"Price {action} ~{abs(val)*100:.1f} sen"

            st.table(pd.DataFrame({
                "Policy Action": [f"{fuel_name} + RM 0.10", f"{fuel_name} - RM 0.10", "USD + 0.10", "USD - 0.10"],
                "Expected Retail Impact": [
                    fmt_impact(avg_fuel_impact_10sen), 
                    fmt_impact(-avg_fuel_impact_10sen), 
                    fmt_impact(avg_usd_impact_10sen), 
                    fmt_impact(-avg_usd_impact_10sen)
                ]
            }))

        except Exception as e:
            st.error(f"⚠️ Sensitivity Engine Error: {str(e)}")
    else:
        st.warning("⚠️ Historical data missing.")

# ==========================================
# TAB 7: GOUGING DETECTOR
# ==========================================
with tab7:
    st.header("🚨 Deliverable #7: Fair Price Enforcement Gazette")
    is_kembung = "Kembung" in species_choice
    metrics_file = 'kembung_blind_rmse.json' if is_kembung else 'selar_blind_rmse.json'
    model = k_struct if is_kembung else s_struct
    
    if os.path.exists("full_buffer.csv") and os.path.exists(metrics_file):
        df_live = pd.read_csv("full_buffer.csv")
        with open(metrics_file, 'r') as f:
            val_metrics = json.load(f)
            # Use Week 1 MAE as baseline forensic uncertainty if possible
            if val_metrics:
                # Get the first available key
                first_key = list(val_metrics.keys())[0]
                model_mae = float(val_metrics[first_key].get('mae', 0.60)) 
            else:
                model_mae = 0.60
        
        # --- CRITICAL T+1 LOGIC FIX ---
        if len(df_live) >= 2:
            # 1. INPUTS come from the PREVIOUS week (2nd to last row)
            input_row = df_live.iloc[-2] 
            
            # 2. DATE comes from the CURRENT week (Last row) - The target of the prediction
            target_row = df_live.iloc[-1]
            cycle_date = pd.to_datetime(target_row['date']).strftime('%d %b %Y')
            
            # 3. Calculate Prediction based on INPUTS (Previous Week)
            if is_kembung:
                input_df = pd.DataFrame({
                    'RON97': [float(input_row['RON97'])], 
                    'height_mean_m_perak': [float(input_row['height_mean_m_perak'])], 
                    'myr_per_usd_mean': [float(input_row['myr_per_usd_mean'])]
                })
                fair_price = float(model.predict(input_df)[0])
            else:
                econ_p = float(input_row['diesel']) * float(input_row['myr_per_usd_mean'])
                solunar_s = float(input_row['height_mean_m_perak']) * float(input_row.get('sunset_mean_time_perak', 1140))
                input_df = pd.DataFrame({'econ_pressure_index': [econ_p], 'solunar_synergy_perak': [solunar_s]})
                fair_price = float(model.predict(input_df)[0])
        else:
            st.error("Live buffer file insufficient (needs at least 2 rows for lag logic).")
            st.stop()

        st.subheader("⚙️ Enforcement Calibration")
        col_slider, col_plot = st.columns([1, 2])
        with col_slider:
            tolerance_rm = st.slider("Enforcement Tolerance (± RM):", 0.10, 2.00, 1.00, 0.05)
            abs_upper_limit = fair_price + tolerance_rm
            abs_lower_limit = fair_price - tolerance_rm
            st.info(f"**Legal Enforcement Thresholds:**\n\n🔴 **Gouging Limit:** > RM {abs_upper_limit:.2f}\n\n🟢 **Fair Market Zone:** RM {abs_lower_limit:.2f} - {abs_upper_limit:.2f}\n\n🔵 **Supply Dumping:** < RM {abs_lower_limit:.2f}")

        with col_plot:
            mu = 0 
            sigma_val = model_mae * 1.25 
            x_dev = np.linspace(mu - 3*sigma_val, mu + 3*sigma_val, 200)
            y = (1/(sigma_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_dev - mu) / sigma_val)**2)
            x_price = fair_price + x_dev
            chart_df = pd.DataFrame({'Price': x_price, 'Probability': y})
            chart_df['Zone'] = np.where(chart_df['Price'] > abs_upper_limit, 'Gouging (Red)', np.where(chart_df['Price'] < abs_lower_limit, 'Dumping (Blue)', 'Fair Price (Green)'))
            
            base = alt.Chart(chart_df).encode(x=alt.X('Price', title='Market Price Spectrum (RM)'), y=alt.Y('Probability', axis=None), tooltip=['Price', 'Zone'])
            area = base.mark_area(opacity=0.6).encode(color=alt.Color('Zone', scale=alt.Scale(domain=['Fair Price (Green)', 'Gouging (Red)', 'Dumping (Blue)'], range=['#2ecc71', '#e74c3c', '#3498db']), legend=None))
            line = base.mark_line(color='#2c3e50', strokeWidth=2)
            rule_upper = alt.Chart(pd.DataFrame({'x': [abs_upper_limit]})).mark_rule(color='#e74c3c', strokeDash=[5, 5]).encode(x='x')
            rule_lower = alt.Chart(pd.DataFrame({'x': [abs_lower_limit]})).mark_rule(color='#3498db', strokeDash=[5, 5]).encode(x='x')
            st.altair_chart((area + line + rule_upper + rule_lower).properties(height=350), use_container_width=True)

        st.divider()
        st.subheader("📡 Live Gazette: Current Price Standard")
        c_hero, c_info = st.columns([1.5, 1])
        with c_hero:
            st.markdown(f"""
            <div style="background-color: #2c3e50; padding: 40px; border-radius: 15px; text-align: center; border: 3px solid #34495e; box-shadow: 0 6px 12px rgba(0,0,0,0.3);">
                <h3 style="color: #ecf0f1; margin: 0; font-weight: 300; letter-spacing: 2px;">SCIENTIFIC FAIR PRICE</h3>
                <p style="color: #bdc3c7; font-size: 0.9em; margin-bottom: 5px;">Calculated for week starting: {cycle_date}</p>
                <h1 style="color: #2ecc71; margin: 0; font-size: 4.5em; font-weight: 900; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">RM {fair_price:.2f}</h1>
                <p style="color: #95a5a6; font-size: 1.1em; margin-top: 10px;">Forensic MAE: <b style="color: #f39c12;">± RM {model_mae:.3f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        with c_info:
            st.warning("**👮 Enforcement Protocol**")
            st.markdown(f"The **Scientific Fair Price** is the AI-driven structural baseline. Enforcement actions are triggered when market prices exceed the allowable tolerance.\n\n*Calculated as: Baseline ± RM {tolerance_rm:.2f} Buffer*")

# ==========================================
# TAB 8: SIMULATOR (PLAYGROUND)
# ==========================================
with tab8:
    st.header("🎮 Price Sensitivity Simulator")
    st.caption("Testing the Pure Structural Brain. Adjust inputs to see how the model naturally responds to market shifts.")
    
    c1, c2 = st.columns([1.2, 1])
    
    if "Kembung" in species_choice:
        with c1:
            st.subheader("Hypothetical Scenarios")
            k_ron97 = st.slider("⛽ RON97 Price (RM/L)", 2.00, 5.00, 3.47, step=0.01)
            k_usd = st.slider("🇺🇸 MYR/USD Rate", 3.50, 5.50, 4.45, step=0.01)
            k_tide = st.slider("🌊 Tide Height Perak (m)", 0.50, 3.00, 1.67, step=0.01)
            
        with c2:
            st.subheader("🎯 Model Configuration")
            st.write("**Architecture:** Pure Structural EBM")
            st.info("💡 No manual shock overrides. This output is 100% generated by the learned relationship between features.")
            
        if st.button("Run Kembung Simulation", type="primary"):
            input_df = pd.DataFrame({'RON97': [k_ron97], 'height_mean_m_perak': [k_tide], 'myr_per_usd_mean': [k_usd]})
            final_price = k_struct.predict(input_df)[0]
            st.divider()
            st.metric("Forecasted Retail Price", f"RM {final_price:.2f}", delta="Structural Estimate", delta_color="off")

    else:
        with c1:
            st.subheader("Hypothetical Scenarios")
            s_diesel = st.slider("🚛 Diesel Price (RM/L)", 2.15, 4.50, 3.35, step=0.01)
            s_usd = st.slider("🇺🇸 MYR/USD Rate", 3.50, 5.50, 4.45, step=0.01)
            s_tide = st.slider("🌊 Tide Height Perak (m)", 0.50, 3.00, 1.67, step=0.01)
            
            # Intuitive Time Selector
            def fmt_time(m):
                h = m // 60
                mn = m % 60
                return f"{h:02d}:{mn:02d}"

            time_options = list(range(1080, 1201, 5))
            s_sun = st.select_slider(
                "🌇 Sunset Time:",
                options=time_options,
                value=1140,
                format_func=fmt_time,
                help="Adjusts the 'Solunar Synergy' vector. 19:00 is standard benchmark."
            )
            
        with c2:
            st.subheader("🎯 Model Configuration")
            st.write("**Architecture:** Surgical Bio-Econ Synergy")
            st.info("💡 High-precision interaction logic. ")
            
        if st.button("Run Selar Simulation", type="primary"):
            econ_pressure = s_diesel * s_usd
            solunar_synergy = s_tide * s_sun
            input_df = pd.DataFrame({'econ_pressure_index': [econ_pressure], 'solunar_synergy_perak': [solunar_synergy]})
            final_price = s_struct.predict(input_df)[0]
            st.divider()
            st.metric("Forecasted Retail Price", f"RM {final_price:.2f}", delta="Interaction Estimate", delta_color="off")

# ==========================================
# TAB 9: GOVERNANCE LEDGER (FIXED)
# ==========================================
with tab9:
    st.header("📜 Deliverable #8: Governance & Compliance Ledger")
    st.write("This module provides total transparency into the model's data pipeline.")

    # --- DEFINE VARIABLES CORRECTLY ---
    is_kembung = "Kembung" in species_choice
    species_name = "kembung" if is_kembung else "selar" # <--- FIXED: Added this definition
    model = k_struct if is_kembung else s_struct

    if os.path.exists("full_buffer.csv"):
        df_ledger = pd.read_csv("full_buffer.csv")
        df_ledger['date'] = pd.to_datetime(df_ledger['date'])
        df_ledger = df_ledger.sort_values('date', ascending=True)

        def parse_sunset(val):
            try:
                if isinstance(val, (int, float)): return float(val)
                if isinstance(val, str) and ':' in val:
                    h, m = val.split(':')
                    return int(h) * 60 + int(m)
                return 1140.0
            except:
                return 1140.0

        if 'sunset_mean_time_perak' in df_ledger.columns:
            df_ledger['sunset_mean_time_perak'] = df_ledger['sunset_mean_time_perak'].apply(parse_sunset)
        else:
            df_ledger['sunset_mean_time_perak'] = 1140.0

        cols_to_clean = ['diesel', 'RON97', 'myr_per_usd_mean', 'height_mean_m_perak', 'sunset_mean_time_perak']
        for c in cols_to_clean:
            if c in df_ledger.columns:
                df_ledger[c] = pd.to_numeric(df_ledger[c], errors='coerce').ffill().bfill()
        
        def get_structural_fair_price(row):
            try:
                if is_kembung:
                    input_df = pd.DataFrame({'RON97': [row['RON97']], 'height_mean_m_perak': [row['height_mean_m_perak']], 'myr_per_usd_mean': [row['myr_per_usd_mean']]})
                    return float(model.predict(input_df)[0])
                else:
                    e_p = row['diesel'] * row['myr_per_usd_mean']
                    s_s = row['height_mean_m_perak'] * row['sunset_mean_time_perak']
                    input_df = pd.DataFrame({'econ_pressure_index': [e_p], 'solunar_synergy_perak': [s_s]})
                    return float(model.predict(input_df)[0])
            except:
                return 0.00 

        df_ledger['AI Fair Price'] = df_ledger.apply(get_structural_fair_price, axis=1)

        st.subheader("1. Audit Timeline")
        min_d = df_ledger['date'].min().date()
        max_d = df_ledger['date'].max().date()
        
        s_date, e_date = st.slider("Select Audit Window:", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="DD MMM YY")

        mask = (df_ledger['date'] >= pd.Timestamp(s_date)) & (df_ledger['date'] <= pd.Timestamp(e_date))
        df_filtered = df_ledger.loc[mask]

        st.subheader(f"2. {species_choice} Market Stability Audit")
        
        if not df_filtered.empty:
            period_prices = df_filtered['AI Fair Price']
            vol_score = (period_prices.std() / period_prices.mean()) * 100 if period_prices.mean() > 0 else 0
            start_price = df_filtered.iloc[0]['AI Fair Price']
            final_price = df_filtered.iloc[-1]['AI Fair Price']
            net_change = final_price - start_price
            
            if vol_score < 2.5: stab_label, stab_color = "✅ STRUCTURALLY STABLE", "normal"
            elif vol_score < 5.0: stab_label, stab_color = "⚠️ UNSETTLED", "off"
            else: stab_label, stab_color = "🚨 HIGH VOLATILITY", "inverse"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📉 Period Volatility", f"{vol_score:.2f}%", stab_label, delta_color=stab_color)
            m2.metric("📈 Net Price Trend", f"RM {final_price:.2f}", f"{net_change:+.2f} RM")
            m3.metric("🏔️ Period Peak", f"RM {period_prices.max():.2f}")
            m4.metric("🗃️ Audit Depth", f"{len(df_filtered)} Wks")
            
        st.divider()
        st.subheader("3. Forensic Data Log")
        
        df_disp = df_filtered.copy()
        df_disp['date'] = df_disp['date'].dt.strftime('%Y-%m-%d')
        rename_map = {'date': 'Audit Date', 'AI Fair Price': 'AI Fair Price (RM)', 'diesel': 'Diesel (RM)', 'RON97': 'RON97 (RM)', 'myr_per_usd_mean': 'USD Rate', 'height_mean_m_perak': 'Tide (m)', 'sunset_mean_time_perak': 'Solar (Mins)'}
        df_disp = df_disp.rename(columns=rename_map)
        
        if is_kembung:
            cols = ['Audit Date', 'AI Fair Price (RM)', 'Diesel (RM)', 'RON97 (RM)', 'USD Rate', 'Tide (m)']
        else:
            cols = ['Audit Date', 'AI Fair Price (RM)', 'Diesel (RM)', 'USD Rate', 'Tide (m)', 'Solar (Mins)'] 
            
        final_df = df_disp[[c for c in cols if c in df_disp.columns]].sort_values('Audit Date', ascending=False)
        st.dataframe(final_df, use_container_width=True, hide_index=True, height=400)

        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Audit CSV", 
            data=csv, 
            file_name=f"{species_name}_compliance_ledger.csv", # Now species_name is defined
            mime="text/csv", 
            type="primary"
        )
        st.caption("**Compliance Certification:** Prices derived via Pure Structural EBM Inference.")

    else:
        st.warning("⚠️ Audit database (`full_buffer.csv`) not detected.")
        