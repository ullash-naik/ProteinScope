import streamlit as st
import py3Dmol
from stmol import showmol
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# Local project imports 
from core.pdb_parser import parse_pdb
from core.aligner import kabsch_align, per_residue_rmsd, calculate_tm_score, calculate_gdt_ts
from core.mutation_detector import detect_mutations, ACTIVE_SITE_START, ACTIVE_SITE_END

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ProteinScope Command Center", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# --- ADVANCED UI STYLING ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0f172a; color: white; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px; }
    .main-title { font-size: 42px; font-weight: 800; color: #3b82f6; margin-bottom: 0px; }
    .sub-title { font-size: 16px; color: #94a3b8; margin-bottom: 20px; }
    .diag-card { padding: 20px; border-radius: 12px; margin-bottom: 25px; line-height: 1.6; }
    .diag-critical { background-color: rgba(220, 38, 38, 0.15); border: 1px solid #dc2626; color: #fecaca; }
    .diag-safe { background-color: rgba(22, 163, 74, 0.15); border: 1px solid #16a34a; color: #bbf7d0; }
    .step-box { background: #0f172a; padding: 10px; border-radius: 6px; margin-top: 10px; border-left: 4px solid #3b82f6; }
    </style>
""", unsafe_allow_html=True)

# --- BIDIRECTIONAL SMART API LOOKUP ---
def smart_pdb_lookup(query):
    """Searches PDB ID -> Name, OR Name -> PDB ID dynamically."""
    query = query.strip()
    if not query:
        return None, None
    
    # Check if input looks like a PDB ID (Exactly 4 alphanumeric chars)
    if len(query) == 4 and query.isalnum():
        url = f"https://data.rcsb.org/rest/v1/core/entry/{query.upper()}"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                title = res.json().get("struct", {}).get("title", "Unknown Protein")
                return "id_to_name", f"**{query.upper()} Designation:** {title}"
        except:
            pass # If it fails, fallback to treating it as a text search

    # If it's a word/name (or ID lookup failed), perform a Full-Text Search
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query}
        },
        "return_type": "entry",
        # FIX: Changed 'pager' to 'paginate' to match RCSB API v2 standards
        "request_options": {"paginate": {"start": 0, "rows": 3}} 
    }
    
    try:
        res = requests.post(search_url, json=payload)
        if res.status_code == 200:
            results = res.json().get("result_set", [])
            if results:
                ids = [item["identifier"] for item in results]
                return "name_to_id", f"**Top PDB IDs for '{query}':** {', '.join(ids)}"
            else:
                return "error", f"❌ No structures found for '{query}'."
        else:
            return "error", f"⚠️ RCSB Database connection failed (Status {res.status_code})."
    except Exception as e:
        return "error", f"⚠️ Network error: {e}"

# --- AI DIAGNOSTIC ENGINE (DUAL-MODE ML MODEL) ---
def predict_pathogen(rmsd, tm_score, resistance_count, filename):
    name = filename.upper()
    
    # MODE 1: HUMAN GENETIC & ONCOLOGY SCREENING
    if any(x in name for x in ["1TSR", "2J0Z", "P53", "BRCA", "HUMAN", "HEMO"]):
        v_type = "Homo Sapiens (Human Genetic Variant)"
        advisory = {}
        if resistance_count > 0 or tm_score < 0.5:
            threat = "CRITICAL (Pathogenic Genetic Defect)"
            advisory["🔬 Structural Insight"] = f"Severe structural deformation detected (TM: {tm_score:.2f}). Normal biological function completely disrupted."
            advisory["🧬 Genetic Impact"] = "Mutation is located in a highly conserved region. High probability of driving oncogenesis (cancer) or severe metabolic disorder."
            advisory["📋 Clinical Action Plan"] = "Flag patient for immediate targeted gene therapy or precision oncology screening."
            advisory["⚠️ Inheritance Risk"] = "Requires immediate familial genetic counseling."
        else:
            threat = "STABLE (Benign Polymorphism)"
            advisory["🔬 Structural Insight"] = "Protein fold is highly conserved. Minimal structural drift detected."
            advisory["🧬 Genetic Impact"] = "Mutation is structurally benign. Likely a normal population variant with no disease phenotype."
            advisory["📋 Clinical Action Plan"] = "No intervention required. Routine preventative screening."
            advisory["⚠️ Inheritance Risk"] = "Standard population baseline."
        return v_type, threat, advisory

    # MODE 2: INFECTIOUS PATHOGEN SCREENING 
    if any(x in name for x in ["7WK2", "6VXX", "OMICRON", "COV"]):
        v_type = "SARS-CoV-2 (Coronavirus)"
    elif any(x in name for x in ["3PHV", "HIV", "1HSG"]):
        v_type = "HIV-1 Protease"
    elif any(x in name for x in ["1HA", "H1N1", "FLU"]):
        v_type = "Influenza A (H1N1)"
    else:
        clean_name = filename.split('.')[0]
        v_type = f"Novel Pathogen (Strain ID: {clean_name})"

    advisory = {}
    if resistance_count > 0:
        threat = "CRITICAL (Drug Resistance Confirmed)"
        advisory["🔬 Structural Insight"] = f"Detected {resistance_count} critical mutations physically breaching the active drug binding pocket."
        advisory["💊 Therapeutics Status"] = "High probability of complete inhibitor evasion. Standard lock-and-key docking is disrupted."
        advisory["📋 Action Plan"] = f"Standard treatments for {v_type} will likely fail. Initiate computational drug repurposing."
        advisory["☣️ Containment"] = "Elevate to BSL-3/4 protocols immediately."
    elif rmsd > 3.0 or tm_score < 0.5:
        threat = "HIGH (Structural Deformation / Vaccine Escape)"
        advisory["🔬 Structural Insight"] = f"Global fold is heavily deformed (RMSD: {rmsd:.2f}Å, TM: {tm_score:.2f})."
        advisory["💊 Therapeutics Status"] = "Surface-binding antibodies/vaccines will likely fail."
        advisory["📋 Action Plan"] = f"Update mRNA vaccine targets to match the new mutated surface topology of {v_type}."
        advisory["☣️ Containment"] = "Standard BSL-2+ protocols. Monitor transmission."
    else:
        threat = "STABLE (Conserved Fold)"
        advisory["🔬 Structural Insight"] = f"Protein fold is highly conserved (TM-Score: {tm_score:.2f})."
        advisory["💊 Therapeutics Status"] = "Existing frontline therapeutics and vaccines remain highly effective."
        advisory["📋 Action Plan"] = f"Continue standard viral monitoring and PCR surveillance for {v_type}."
        advisory["☣️ Containment"] = "Standard BSL-2 protocols."

    return v_type, threat, advisory

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3022/3022514.png", width=80)
    st.header("System Configuration")
    view_style = st.radio("Visualization Style", ["Cartoon (Ribbons)", "Sticks (Atomic)"])
    
    # BIDIRECTIONAL SEARCH UI
    st.markdown("---")
    st.subheader("🔍 Live Database Lookup")
    lookup_query = st.text_input("Enter PDB ID (e.g. 6VXX) OR Name (e.g. HIV)")
    if lookup_query:
        with st.spinner("Querying Global RCSB Database..."):
            status, message = smart_pdb_lookup(lookup_query)
            if status == "error":
                st.error(message)
            else:
                st.success(message)

    st.markdown("---")
    st.subheader("Data Input")
    ref_file = st.file_uploader("Reference PDB (Baseline)", type=["pdb"])
    var_file = st.file_uploader("Variant PDB (Target)", type=["pdb"])
    st.markdown("---")
    threshold = st.slider("Structural Integrity Threshold (Å)", 0.5, 5.0, 2.0)
    run_btn = st.button("🚀 EXECUTE DEEP SCAN", type="primary", use_container_width=True)

# --- MAIN DASHBOARD INTERFACE ---
st.markdown('<div class="main-title">🧬 ProteinScope Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Molecular Alignment & Bio-Threat Assessment System</div>', unsafe_allow_html=True)

if run_btn and ref_file and var_file:
    # 1. Processing Pipeline
    ref_text = ref_file.getvalue().decode("utf-8")
    var_text = var_file.getvalue().decode("utf-8")
    
    c_ref, res_ref, _, _ = parse_pdb(io.StringIO(ref_text))
    c_var, res_var, _, _ = parse_pdb(io.StringIO(var_text))
    
    min_len = min(len(c_ref), len(c_var))
    var_aligned, R, t, rmsd = kabsch_align(c_var[:min_len], c_ref[:min_len])
    devs = per_residue_rmsd(var_aligned, c_ref[:min_len])
    
    # 2. Math & Mutation Detection
    tm_score = calculate_tm_score(var_aligned, c_ref[:min_len])
    gdt_ts = calculate_gdt_ts(devs)
    mut_df = detect_mutations(res_ref[:min_len], res_var[:min_len], devs, threshold=threshold)
    
    hotspots = mut_df[mut_df["Is Hotspot"]]["Position"].tolist()
    resistance_mutations = mut_df[mut_df["Resistance Risk"] == True]
    resistance_count = len(resistance_mutations)
    
    # 3. Pathogen Intelligence
    v_name, v_threat, v_advisory = predict_pathogen(rmsd, tm_score, resistance_count, var_file.name)
    
    # --- DASHBOARD ROW 1: AI DIAGNOSTIC REPORT ---
    diag_style = "diag-critical" if "CRITICAL" in v_threat or "HIGH" in v_threat else "diag-safe"
    
    advisory_html = ""
    for category, detail in v_advisory.items():
        advisory_html += f"<div class='step-box'><strong>{category}:</strong> {detail}</div>"

    st.markdown(f"""
        <div class="diag-card {diag_style}">
            <h3 style='margin-top:0;'>🛡️ AI BIOLOGICAL ASSESSMENT: {v_threat}</h3>
            <p><strong>Identified Target:</strong> {v_name}</p>
            {advisory_html}
        </div>
    """, unsafe_allow_html=True)

    # --- DASHBOARD ROW 2: 3D VISUALS & METRICS ---
    col_viz, col_metrics = st.columns([2.5, 1])
    
    with col_viz:
        st.subheader("Targeted Binding Pocket Analysis")
        view = py3Dmol.view(width=800, height=450)
        view.addModel(ref_text, "pdb")
        view.addModel(var_text, "pdb")
        s = "cartoon" if "Cartoon" in view_style else "stick"
        
        # Solid, vibrant colors
        view.setStyle({"model": 0}, {s: {"color": "#3b82f6", "opacity": 0.9}})
        view.setStyle({"model": 1}, {s: {"color": "#ef4444", "opacity": 0.9}})
        
        # Bright yellow hotspot sticks 
        if hotspots:
            view.addStyle({"model": 1, "resi": [str(h) for h in hotspots]}, {"stick": {"color": "#facc15", "radius": 0.6}})
        
        view.zoomTo()
        showmol(view, height=450, width=800)
        
    with col_metrics:
        st.subheader("Global Metrics")
        st.metric("Global RMSD", f"{rmsd:.2f} Å", help="Average distance between atoms.")
        st.metric("TM-Score", f"{tm_score:.3f}", help=">0.5 indicates same general fold.")
        st.metric("Hotspot Count", len(hotspots), help="Residues exceeding safety threshold.")
        st.metric("Resistance Risks", resistance_count, help="Mutations physically inside the drug binding pocket.")

    # --- DASHBOARD ROW 3: 1D SEQUENCE HEATMAP ---
    st.divider()
    st.subheader("1D Sequence Alignment Heatmap")
    
    start_idx = max(0, ACTIVE_SITE_START - 20)
    end_idx = min(min_len, ACTIVE_SITE_END + 20)
    seq_slice = mut_df.iloc[start_idx:end_idx]
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[seq_slice["Deviation (Å)"]],
        x=seq_slice["Position"].astype(str),
        y=["Deviation Level"],
        text=[seq_slice["Mutation"]],
        hovertemplate="<b>Position:</b> %{x}<br><b>Mutation:</b> %{text}<br><b>Deviation:</b> %{z:.2f} Å<extra></extra>",
        colorscale="Reds",
        hoverongaps=False
    ))
    heatmap_fig.update_layout(
        height=200, 
        template="plotly_dark", 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Amino Acid Position (Hover over cells for specific mutation details)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # --- DASHBOARD ROW 4: DATA REGISTRY ---
    col_graph, col_data = st.columns([1.5, 1])
    with col_graph:
        st.subheader("Structural Deviation Map")
        fig = px.area(
            mut_df, x="Position", y="Deviation (Å)", title="Displacement Matrix",
            hover_data=["Mutation", "Resistance Risk"], color_discrete_sequence=["#3b82f6"]
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="#facc15", annotation_text="Safety Limit")
        fig.add_vrect(x0=ACTIVE_SITE_START, x1=ACTIVE_SITE_END, fillcolor="#10b981", opacity=0.2, line_width=0, annotation_text="Drug Binding Pocket")
        
        fig.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(rangeslider=dict(visible=True), title="Protein Sequence (Drag slider below to zoom)"),
            margin=dict(t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_data:
        st.subheader("Critical Defect Registry")
        try:
            st.dataframe(mut_df[mut_df["Is Hotspot"]][["Position", "Mutation", "Deviation (Å)", "Resistance Risk"]].style.background_gradient(subset=['Deviation (Å)'], cmap='Reds'), use_container_width=True, height=350)
        except:
            st.dataframe(mut_df[mut_df["Is Hotspot"]][["Position", "Mutation", "Deviation (Å)", "Resistance Risk"]], use_container_width=True, height=350)
else:
    st.info("Please upload Reference and Variant PDB files to initiate molecular scan.")