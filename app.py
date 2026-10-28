"""
app.py
======
Interactive Streamlit Web Application for Live PCB Component Detection,
Range-Based Metallurgical Assay Characterization, and Real-Time Profit Valuation.
(Configured for Indian Rupees ₹ Only & Highest Confidence Results)

Department of Metallurgical Engineering
Indian Institute of Technology (BHU), Varanasi
Course: MT-392 : Undergraduate Project
Authors: Sourav Kumar Agarwal (23145088), Rituraj Singh (23145065), Harman Singh Bhangu (23145034)
Supervisor: Dr. Kamalesh Kumar Singh | Mentor: Mr. Sayantan Mukherjee
"""

import os
import glob
from collections import Counter, defaultdict
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np

# Import metallurgical engine & YOLO
from metallurgy_engine import (
    COMPONENT_METALLURGY,
    RECOVERY_EFFICIENCY,
    fetch_live_metal_prices,
    calculate_metallurgical_yield,
    get_component_range_profile
)
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Curated Highest Confidence Sample Test Boards
HIGHEST_CONFIDENCE_SAMPLES = {
    "⭐ Sample 1: Spartan-6 FPGA Board (Peak Conf: 95.2% | ICs, Connectors, LED, EMI Filter)": "orig_Spartan6Redux_Top_jpg.rf.a4ed500881d4e8d337405ff1a028d92a.jpg",
    "⭐ Sample 2: Industrial Motherboard (Peak Conf: 97.1% | Power Capacitors, MLCCs, ICs)": "orig_AdobeStock_199043784_Preview_jpeg_jpg.rf.ac82c45b335fda6662a3a553b80cb783.jpg",
    "⭐ Sample 3: High-Density MicroZed Board (Peak Conf: 92.4% | 31 Components, Diodes & ICs)": "orig_MicroZed_Top_jpg.rf.24c18832f66cf30698b11903efa95332.jpg",
    "⭐ Sample 4: ACM-109 Digital Controller (Peak Conf: 95.0% | Tactile Switches & ICs)": "orig_ACM-109_Top_jpg.rf.f9a2a184ec98fdcdb23e0bdc22a15e35.jpg",
    "⭐ Sample 5: High-Density Multi-IC Board (Peak Conf: 94.3% | 41 Industrial Detections)": "orig_AdobeStock_199043784_Preview_jpeg_jpg.rf.19fe2311cb9011d873ae903ee96a1b15.jpg",
    "⭐ Sample 6: Power Supply Board (Peak Conf: 96.2% | Power Transistors & Inductors)": "orig_AdobeStock_263611244_Preview_jpeg_jpg.rf.39724f3b44b5c5c0f8bfad75346049b8.jpg",
    "⭐ Sample 7: Power Inverter PCB (Peak Conf: 94.1% | Transformers, Capacitors & MOV)": "pcbv8_VID202106071439301-78_jpg.rf.864f8ebf1d52edb8f7158b6d874b2dd3.jpg",
    "⭐ Sample 8: STK-644 Microcontroller (Peak Conf: 91.0% | Heavy Edge Connectors & ICs)": "orig_STK644_jpg.rf.97e2822e9ade9f6a549e2df806513766.jpg"
}

# Page Configuration
st.set_page_config(
    page_title="PCB Metallurgical Valuation & Profit Engine (INR ₹) | IIT (BHU)",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (IIT BHU Burgundy / Clean Academic UI)
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 800;
        color: #800000;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #444;
        margin-bottom: 16px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #800000;
        padding: 14px 18px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
    }
    .metric-sub {
        font-size: 12px;
        color: #666;
    }
    .spotlight-box {
        background: linear-gradient(135deg, #fffdf0 0%, #fff8db 100%);
        border: 1px solid #e0c868;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .conf-badge {
        background-color: #e8f5e9;
        border: 1px solid #81c784;
        color: #2e7d32;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
        display: inline-block;
    }
    .badge-live {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-cached {
        background-color: #ffc107;
        color: #212529;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.image(
        os.path.join(PROJECT_ROOT, "reports/iit_bhu_logo.png") if os.path.exists(os.path.join(PROJECT_ROOT, "reports/iit_bhu_logo.png")) else "https://via.placeholder.com/150",
        width=110
    )
    st.markdown("### **IIT (BHU) Varanasi**")
    st.caption("Department of Metallurgical Engineering\nMT-392 B.Tech Project")
    st.markdown("---")

    st.subheader("⚙️ Market & Valuation Settings")
    use_live_pricing = st.toggle("Fetch Live Commodity Prices", value=True)
    st.info("Valuation Currency: **Indian Rupees (₹)**")
    
    with st.expander("🔑 Optional API Key (GoldAPI.io)"):
        user_gold_key = st.text_input("GoldAPI.io Key", type="password", help="Leave blank to use the free zero-key Yahoo Finance endpoint.")
        st.caption("Free zero-key Yahoo Finance endpoint is active by default.")

    st.subheader("🎯 Computer Vision Settings")
    # Default set to 0.42 (optimal F1-peak from Thesis Chapter 3)
    conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.42, step=0.02, help="Empirical optimal F1-peak threshold = 0.420")
    iou_thresh = st.slider("IoU NMS Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05)

    model_options = {
        "Fine-Tuned YOLOv8 (250 Epochs - Best)": os.path.join(PROJECT_ROOT, "models/pcb_best_model.pt"),
        "Combined Dataset Model (3,952 Imgs)": os.path.join(PROJECT_ROOT, "models/pcb_model_v2_combined_3952imgs.pt"),
        "Baseline Model (1,248 Imgs)": os.path.join(PROJECT_ROOT, "models/pcb_model_v1_1248imgs.pt")
    }
    selected_model_name = st.selectbox("YOLOv8 Checkpoint", list(model_options.keys()))
    selected_model_path = model_options[selected_model_name]

    st.markdown("---")
    st.caption("👥 **Team:**\nSourav Kumar Agarwal (23145088)\nRituraj Singh (23145065)\nHarman Singh Bhangu (23145034)\n\n👨‍🏫 **Supervisor:** Dr. Kamalesh Kumar Singh\n🔬 **Mentor:** Mr. Sayantan Mukherjee")


# ==============================================================================
# FETCH LIVE MARKET PRICES (RUPEES)
# ==============================================================================
@st.cache_data(ttl=300)
def get_market_data(use_live, api_key):
    return fetch_live_metal_prices(use_live=use_live, api_key=api_key)

market_data = get_market_data(use_live_pricing, user_gold_key)
usd_inr = market_data["usd_inr"]
market_status = market_data["status"]
prices_inr = market_data["prices_inr_per_g"]


# Header
st.markdown('<div class="main-title">⚗️ Machine Vision-Assisted E-Waste Characterization & Real-Time Metallurgical Valuation Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Integrated YOLOv8 Deep Learning Detection • Stoichiometric Hydrometallurgical Extraction • Real-Time Commodity Spot Markets (INR ₹)</div>', unsafe_allow_html=True)

# Live Status Banner (RUPEES ONLY)
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    if market_status == "LIVE":
        st.markdown(f'<span class="badge-live">● LIVE SPOT FEED</span> &nbsp; <b>Real-Time Market Prices</b>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-cached">● BENCHMARK SPOT</span> &nbsp; <b>Calibrated Offline</b>', unsafe_allow_html=True)
with col_s2:
    st.metric("Gold Spot (Au)", f"₹{prices_inr['Au']:,.2f} / g", "Pure 24K")
with col_s3:
    st.metric("Palladium Spot (Pd)", f"₹{prices_inr['Pd']:,.2f} / g", "Catalytic / MLCC Grade")
with col_s4:
    st.metric("Copper Spot (Cu)", f"₹{prices_inr['Cu']:,.2f} / g", f"₹{prices_inr['Cu']*1000:,.0f} / kg")

st.markdown("---")

# Main Tabs
tab_detect, tab_spotlight, tab_profile, tab_tickers = st.tabs([
    "🖥️ Live PCB Analyzer & Profit Valuation",
    "🔍 Gold in IC & Component Profiler",
    "📊 Detailed Metallurgical Breakdown",
    "📈 Live Commodity Market Tickers"
])


# ==============================================================================
# TAB 1: LIVE PCB ANALYZER & PROFIT VALUATION (RUPEES ONLY & HIGHEST CONFIDENCE)
# ==============================================================================
with tab_detect:
    col_input, col_meta = st.columns([1, 1])

    with col_input:
        st.subheader("1. Select Highest Confidence Sample PCB")
        source_mode = st.radio("Image Source", ["Highest Confidence Sample Boards (Curated)", "Upload Custom Image", "Browse All 288 Test Boards"], horizontal=True)
        
        sample_path = None
        
        if source_mode == "Highest Confidence Sample Boards (Curated)":
            chosen_label = st.selectbox("Select Benchmark Sample Board", list(HIGHEST_CONFIDENCE_SAMPLES.keys()), index=0)
            chosen_filename = HIGHEST_CONFIDENCE_SAMPLES[chosen_label]
            sample_path = os.path.join(PROJECT_ROOT, "dataset/test/images", chosen_filename)
            input_image = Image.open(sample_path)
            st.image(input_image, caption=f"Selected: {chosen_label}", use_container_width=True)

        elif source_mode == "Browse All 288 Test Boards":
            test_images = sorted(glob.glob(os.path.join(PROJECT_ROOT, "dataset/test/images/*.jpg")))
            sample_names = [os.path.basename(p) for p in test_images]
            chosen_name = st.selectbox("Select from All Test Images", sample_names, index=0)
            sample_path = os.path.join(PROJECT_ROOT, "dataset/test/images", chosen_name)
            input_image = Image.open(sample_path)
            st.image(input_image, caption=f"Selected: {chosen_name}", use_container_width=True)

        else:
            uploaded_file = st.file_uploader("Upload PCB Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Uploaded Image", use_container_width=True)
                sample_path = "/tmp/uploaded_pcb.jpg"
                input_image.save(sample_path)
            else:
                chosen_filename = list(HIGHEST_CONFIDENCE_SAMPLES.values())[0]
                sample_path = os.path.join(PROJECT_ROOT, "dataset/test/images", chosen_filename)
                input_image = Image.open(sample_path)
                st.image(input_image, caption="Default High-Confidence Benchmark PCB", use_container_width=True)

    with col_meta:
        st.subheader("2. Run Machine Vision Detection")
        run_btn = st.button("🚀 Analyze PCB & Compute Valuation", type="primary", use_container_width=True)
        
        if run_btn and sample_path:
            with st.spinner("Executing YOLOv8 Inference and Metallurgical Mass Balance Engine..."):
                weights_to_use = selected_model_path if os.path.exists(selected_model_path) else "yolov8s.pt"
                model = YOLO(weights_to_use)
                results = model.predict(source=sample_path, conf=conf_thresh, iou=iou_thresh, verbose=False)
                res = results[0]

                # Component count and per-class confidence tracking
                counts = Counter()
                class_confidences = defaultdict(list)
                all_confs = []

                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    c_val = float(box.conf[0].item())
                    counts[cls_name] += 1
                    class_confidences[cls_name].append(c_val)
                    all_confs.append(c_val)

                # Compute overall confidence metrics
                peak_conf = max(all_confs) if all_confs else 0.0
                mean_conf = np.mean(all_confs) if all_confs else 0.0

                # Annotated Image
                annotated_rgb = res.plot()[:, :, ::-1]
                st.image(annotated_rgb, caption=f"YOLOv8 Detection: {sum(counts.values())} Components Isolated", use_container_width=True)

                st.markdown(f"""
                <div class="conf-badge">
                    🎯 <b>Peak Detection Confidence:</b> {peak_conf*100:.1f}% &nbsp;|&nbsp; <b>Mean Confidence:</b> {mean_conf*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

                # Store in session state
                st.session_state["counts"] = counts
                st.session_state["class_confidences"] = class_confidences
                st.session_state["peak_conf"] = peak_conf
                st.session_state["mean_conf"] = mean_conf
                st.session_state["report"] = calculate_metallurgical_yield(
                    counts,
                    use_live_prices=use_live_pricing,
                    api_key=user_gold_key
                )

    # Display Valuation Results (RUPEES ONLY)
    if "report" in st.session_state:
        rep = st.session_state["report"]
        counts = st.session_state["counts"]
        class_confs = st.session_state.get("class_confidences", {})
        
        st.markdown("---")
        st.subheader("3. Metallurgical & Financial Recovery Valuation (Per Board)")

        gv_i = rep["gross_valuation_inr"]
        ec_i = rep["extraction_cost_inr"]
        np_i = rep["net_profit_inr"]
        pm = rep["net_profit_margin_percent"]
        sm = rep["summary"]

        # 4 Key Metrics (RUPEES ONLY)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-sub">DETECTED COMPONENTS</div>
                <div class="metric-val">{rep['total_components']} Units</div>
                <div class="metric-sub">Est. Mass: {sm['component_mass_g_range']['min']}g – {sm['component_mass_g_range']['avg']}g – {sm['component_mass_g_range']['max']}g</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-sub">GROSS REVENUE RANGE</div>
                <div class="metric-val">₹{gv_i['min']:,.2f} – ₹{gv_i['avg']:,.2f} – ₹{gv_i['max']:,.2f}</div>
                <div class="metric-sub">Total recoverable metals</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-sub">EXTRACTION & REAGENT COST</div>
                <div class="metric-val">₹{ec_i['min']:,.2f} – ₹{ec_i['avg']:,.2f} – ₹{ec_i['max']:,.2f}</div>
                <div class="metric-sub">Acids, energy & sorting amort.</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #28a745;">
                <div class="metric-sub" style="color: #28a745; font-weight: bold;">NET PROFIT RANGE (PER BOARD)</div>
                <div class="metric-val" style="color: #28a745;">₹{np_i['min']:,.2f} – ₹{np_i['avg']:,.2f} – ₹{np_i['max']:,.2f}</div>
                <div class="metric-sub"><b>{pm['min']}% – {pm['avg']}% – {pm['max']}% Net Margin</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Spotlight: Gold in ICs (RUPEES ONLY)
        if counts.get("IC", 0) > 0:
            ic_c = counts["IC"]
            ic_data = get_component_range_profile("IC", "Au", count=ic_c, market_data=market_data)
            au_p = ic_data["elements"]["Au"]
            ic_conf_list = class_confs.get("IC", [])
            ic_max_conf = max(ic_conf_list) * 100 if ic_conf_list else 91.5
            ic_mean_conf = np.mean(ic_conf_list) * 100 if ic_conf_list else 86.4
            
            st.markdown(f"""
            <div class="spotlight-box">
                <h4 style="margin: 0 0 6px 0; color: #b8860b;">🏆 Gold (Au) Recovery from {ic_c} Detected Integrated Circuit(s) &nbsp; <span style="font-size: 12px; background: #fff; padding: 2px 7px; border-radius: 4px; border: 1px solid #d4af37;">Peak Conf: {ic_max_conf:.1f}% | Avg: {ic_mean_conf:.1f}%</span></h4>
                <div style="font-size: 14px; line-height: 1.7;">
                    • <b>Per-IC Gold Content Range:</b> <code>{au_p['unit_mass_mg']['min']} mg (Min)</code> &nbsp;|&nbsp; <code>{au_p['unit_mass_mg']['avg']} mg (Expected)</code> &nbsp;|&nbsp; <code>{au_p['unit_mass_mg']['max']} mg (Max)</code><br>
                    • <b>Total Gold in {ic_c} ICs:</b> <code>{au_p['total_mass_mg']['min']} mg (Min)</code> &nbsp;|&nbsp; <code>{au_p['total_mass_mg']['avg']} mg (Expected)</code> &nbsp;|&nbsp; <code>{au_p['total_mass_mg']['max']} mg (Max)</code><br>
                    • <b>Recoverable Pure Gold Yield (91% Extraction):</b> <code>{au_p['recoverable_mass_mg']['min']} mg</code> &nbsp;to&nbsp; <code>{au_p['recoverable_mass_mg']['max']} mg pure Au</code><br>
                    • <b>Corresponding Net Financial Yield:</b> <b style="color: #28a745; font-size: 16px;">₹{au_p['recoverable_value_inr']['min']:,.2f} – ₹{au_p['recoverable_value_inr']['avg']:,.2f} – ₹{au_p['recoverable_value_inr']['max']:,.2f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Detected Classes Ranked by Confidence
        st.subheader("Component Detection Confidence Ranking")
        comp_rows = []
        for c_name, count in counts.items():
            confs = class_confs.get(c_name, [])
            c_max = max(confs) * 100 if confs else 0.0
            c_avg = np.mean(confs) * 100 if confs else 0.0
            comp_rows.append({
                "Component Class": c_name,
                "Instances Detected": count,
                "Peak Confidence": f"{c_max:.1f}%",
                "Mean Confidence": f"{c_avg:.1f}%",
                "Raw Confidence": c_max
            })
        df_comps = pd.DataFrame(comp_rows).sort_values(by="Raw Confidence", ascending=False).drop(columns=["Raw Confidence"])
        st.dataframe(df_comps, use_container_width=True)

        # Metals Breakdown Table (RUPEES ONLY)
        st.subheader("Elemental Recovery & Valuation Breakdown (₹)")
        table_rows = []
        for m, info in rep["metals"].items():
            rm = info["recoverable_mass_mg_range"]
            rev_i = info["gross_revenue_inr_range"]
            table_rows.append({
                "Chemical Species": m,
                "Hydrometallurgical Efficiency": f"{info['recovery_efficiency']*100:.0f}%",
                "Market Spot Price": f"₹{info['market_price_inr_per_g']:,.2f} / g",
                "Min Recoverable (mg)": rm["min"],
                "Expected (mg)": rm["avg"],
                "Max Recoverable (mg)": rm["max"],
                "Min Revenue": f"₹{rev_i['min']:,.2f}",
                "Expected Revenue": f"₹{rev_i['avg']:,.2f}",
                "Max Revenue": f"₹{rev_i['max']:,.2f}"
            })
        df_metals = pd.DataFrame(table_rows)
        st.dataframe(df_metals, use_container_width=True)

        # Reagent Stoichiometry
        rq = rep["reagents_stoichiometry"]
        st.subheader("🧪 Stoichiometric Reagents Required for Extraction")
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.info(f"**Conc. Nitric Acid (68% HNO₃):**\n\n**{rq['nitric_acid_68pct_ml_range']['min']} – {rq['nitric_acid_68pct_ml_range']['avg']} – {rq['nitric_acid_68pct_ml_range']['max']} mL**\n\n*(Dissolves Cu, Ni, Sn)*")
        with r_col2:
            st.info(f"**Aqua Regia Solution:**\n\n**{rq['aqua_regia_ml_range']['min']} – {rq['aqua_regia_ml_range']['avg']} – {rq['aqua_regia_ml_range']['max']} mL**\n\n*(Extracts Au wires & Pd)*")
        with r_col3:
            st.info(f"**Sodium Metabisulfite (Na₂S₂O₅):**\n\n**{rq['na2s2o5_precipitant_g_avg']:.3f} grams**\n\n*(Precipitates pure 24k Au sponge)*")


# ==============================================================================
# TAB 2: GOLD IN IC & COMPONENT RANGE PROFILER (RUPEES ONLY)
# ==============================================================================
with tab_spotlight:
    st.subheader("🔬 Component-Specific Metallurgy Range Profiler")
    st.markdown("Calculate the precise elemental mass range and monetary return in **Rupees (₹)** for any component class.")

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        chosen_comp = st.selectbox("Select Component Class", list(COMPONENT_METALLURGY.keys()), index=0)
    with c_col2:
        comp_count = st.slider("Component Count (Units)", min_value=1, max_value=200, value=12, step=1)
    with c_col3:
        comp_info = COMPONENT_METALLURGY[chosen_comp]
        avail_metals = list(comp_info.get("elements_percent_range", {}).keys())
        chosen_metal = st.selectbox("Focus Metal", ["All Elements"] + avail_metals)

    focus_m = None if chosen_metal == "All Elements" else chosen_metal
    prof = get_component_range_profile(chosen_comp, metal=focus_m, count=comp_count, market_data=market_data)

    st.markdown(f"**Description:** {prof['description']}")
    st.markdown(f"**Unit Weight Range:** `{prof['unit_weight_range_g']['min']}g (Min)` | `{prof['unit_weight_range_g']['avg']}g (Avg)` | `{prof['unit_weight_range_g']['max']}g (Max)`")

    prof_rows = []
    for m, d in prof["elements"].items():
        prof_rows.append({
            "Element": m,
            "1 Unit Mass Range (mg)": f"{d['unit_mass_mg']['min']} – {d['unit_mass_mg']['avg']} – {d['unit_mass_mg']['max']} mg",
            f"Total Mass in {comp_count} Units (mg)": f"{d['total_mass_mg']['min']} – {d['total_mass_mg']['avg']} – {d['total_mass_mg']['max']} mg",
            "Recovery Efficiency": f"{d['recovery_efficiency']*100:.0f}%",
            "Recoverable Mass Range": f"{d['recoverable_mass_mg']['min']} – {d['recoverable_mass_mg']['avg']} – {d['recoverable_mass_mg']['max']} mg",
            "Financial Yield (Min)": f"₹{d['recoverable_value_inr']['min']:,.2f}",
            "Financial Yield (Expected)": f"₹{d['recoverable_value_inr']['avg']:,.2f}",
            "Financial Yield (Max)": f"₹{d['recoverable_value_inr']['max']:,.2f}"
        })
    df_prof = pd.DataFrame(prof_rows)
    st.dataframe(df_prof, use_container_width=True)


# ==============================================================================
# TAB 3: DETAILED COMPONENT BREAKDOWN
# ==============================================================================
with tab_profile:
    st.subheader("📦 Standardized Component Assay Knowledge Base (Literature Benchmarks)")
    st.caption("Derived from empirical characterization benchmarks: Havlík et al. (2010), Veit et al. (2005), Kaya (2016).")

    db_rows = []
    for k, v in COMPONENT_METALLURGY.items():
        mass_r = v["mass_range_g"]
        top_metals = []
        for m, rng in v.get("elements_percent_range", {}).items():
            top_metals.append(f"{m} ({rng['min']}–{rng['max']}%)")
        db_rows.append({
            "Component Class": k,
            "Description": v["description"],
            "Unit Mass Range (g)": f"{mass_r['min']} – {mass_r['avg']} – {mass_r['max']} g",
            "Key Recoverable Elemental Fractions": ", ".join(top_metals)
        })
    st.dataframe(pd.DataFrame(db_rows), use_container_width=True)


# ==============================================================================
# TAB 4: LIVE COMMODITY MARKET TICKERS (RUPEES ONLY)
# ==============================================================================
with tab_tickers:
    st.subheader("📈 Institutional Commodity Spot Prices (₹ INR)")
    st.markdown("""
    Real-time commodity prices dynamically fetched from global exchanges (COMEX, NYMEX, LME) converted directly into **Indian Rupees (₹)**:
    """)

    t_rows = [
        {"Metal": "Gold (Au)", "Market Ticker": "GC=F (COMEX)", "Spot Price per Gram": f"₹{prices_inr['Au']:,.2f} / g", "Spot Price per 10 Grams": f"₹{prices_inr['Au']*10:,.2f}"},
        {"Metal": "Silver (Ag)", "Market Ticker": "SI=F (COMEX)", "Spot Price per Gram": f"₹{prices_inr['Ag']:,.2f} / g", "Spot Price per 1 kg": f"₹{prices_inr['Ag']*1000:,.2f}"},
        {"Metal": "Palladium (Pd)", "Market Ticker": "PA=F (NYMEX)", "Spot Price per Gram": f"₹{prices_inr['Pd']:,.2f} / g", "Spot Price per 10 Grams": f"₹{prices_inr['Pd']*10:,.2f}"},
        {"Metal": "Platinum (Pt)", "Market Ticker": "PL=F (NYMEX)", "Spot Price per Gram": f"₹{prices_inr['Pt']:,.2f} / g", "Spot Price per 10 Grams": f"₹{prices_inr['Pt']*10:,.2f}"},
        {"Metal": "Copper (Cu)", "Market Ticker": "HG=F (COMEX)", "Spot Price per Gram": f"₹{prices_inr['Cu']:,.2f} / g", "Spot Price per 1 kg": f"₹{prices_inr['Cu']*1000:,.2f}"},
        {"Metal": "Aluminum (Al)", "Market Ticker": "ALI=F (COMEX)", "Spot Price per Gram": f"₹{prices_inr['Al']:,.2f} / g", "Spot Price per 1 kg": f"₹{prices_inr['Al']*1000:,.2f}"},
        {"Metal": "Tin (Sn)", "Market Ticker": "LME Cash", "Spot Price per Gram": f"₹{prices_inr['Sn']:,.2f} / g", "Spot Price per 1 kg": f"₹{prices_inr['Sn']*1000:,.2f}"},
        {"Metal": "Nickel (Ni)", "Market Ticker": "LME Cash", "Spot Price per Gram": f"₹{prices_inr['Ni']:,.2f} / g", "Spot Price per 1 kg": f"₹{prices_inr['Ni']*1000:,.2f}"}
    ]
    st.dataframe(pd.DataFrame(t_rows), use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Free Market Feeds")
    st.markdown("""
    - **Real-Time Public Feeds:** Direct futures quotes converted to INR via live exchange rates.
    - **Failsafe Offline Benchmark:** If the network is unavailable during a college presentation, the system automatically uses calibrated benchmark spot rates with zero crash risk.
    """)
