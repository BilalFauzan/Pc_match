"""
Automated Static-Dynamic Initial Water Saturation (Sw) Matching & Pc-Sw Optimization
=====================================================================================
Interactive Streamlit Application for Reservoir Engineers and Geoscientists.
Supports arbitrary Corner Point Grids, Multi-Region Compartments (EQLNUM),
and SCAL Rock Type Curve Calibration for Petrel and tNavigator / ECLIPSE.
"""

import os
import sys
import io
import time
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure workspace directory is in sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

import pc_matching as pcm


def scan_grid_rock_types(grid_filepath, kw_rt, kw_depth=None, kw_eql=None, kw_poro=None, min_poro=0.0):
    """
    Rapidly scan grid file to extract available rock types, cell counts, depth ranges, and zone numbers.
    """
    try:
        available_kw = [kw for _, kw in pcm.FastGRDECLParser.scan_keywords(grid_filepath)]
        if kw_rt not in available_kw:
            return None, f"Keyword Rock Type '{kw_rt}' tidak ditemukan di dalam file grid."

        rt_arr = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_rt)
        tot_cells = len(rt_arr)

        poro_arr = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_poro, total_cells=tot_cells) if (kw_poro and kw_poro in available_kw) else None
        depth_arr = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_depth, total_cells=tot_cells) if (kw_depth and kw_depth in available_kw) else None
        eql_arr = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_eql, total_cells=tot_cells, dtype=np.int32) if (kw_eql and kw_eql in available_kw) else None

        u_rts = sorted([int(r) for r in np.unique(rt_arr) if r > 0])
        rt_info = {}
        active_rts = []
        for r in u_rts:
            mask_r = (rt_arr == r)
            n_cells = int(np.sum(mask_r))

            pay_mask = mask_r
            if poro_arr is not None and min_poro > 0:
                pay_mask = pay_mask & (poro_arr >= min_poro)
            n_pay = int(np.sum(pay_mask))

            tvd_min, tvd_max = 0.0, 0.0
            if depth_arr is not None:
                d_sub = depth_arr[pay_mask & (depth_arr > 0)] if n_pay > 0 else depth_arr[mask_r & (depth_arr > 0)]
                if len(d_sub) > 0:
                    tvd_min, tvd_max = float(np.min(d_sub)), float(np.max(d_sub))

            zones = []
            if eql_arr is not None:
                z_sub = eql_arr[pay_mask & (eql_arr > 0)] if n_pay > 0 else eql_arr[mask_r & (eql_arr > 0)]
                if len(z_sub) > 0:
                    zones = sorted([int(z) for z in np.unique(z_sub)])

            is_active = (n_pay > 0)
            if is_active:
                active_rts.append(r)

            rt_info[r] = {
                'total_cells': n_cells,
                'pay_cells': n_pay,
                'tvd_min': tvd_min,
                'tvd_max': tvd_max,
                'zones': zones,
                'is_active': is_active
            }

        if not active_rts:
            active_rts = u_rts

        return {
            'all_rts': u_rts,
            'active_rts': active_rts,
            'rt_info': rt_info,
            'available_kw': available_kw,
            'tot_cells': tot_cells
        }, None
    except Exception as e:
        return None, str(e)


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pc-Sw Matching & Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'grid_data' not in st.session_state:
    st.session_state['grid_data'] = None
if 'scal_manager' not in st.session_state:
    st.session_state['scal_manager'] = None
if 'opt_results' not in st.session_state:
    st.session_state['opt_results'] = None
if 'equilibrator' not in st.session_state:
    st.session_state['equilibrator'] = None


# -----------------------------------------------------------------------------
# SIDEBAR: DATA UPLOAD & CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/oil-rig.png", width=64)
st.sidebar.title("Input & Configuration")

temp_dir = os.path.join(tempfile.gettempdir(), "pc_matching_uploads")
os.makedirs(temp_dir, exist_ok=True)

# Grid file and keyword variables
grid_filepath = None
sol_filepath = None
props_filepath = None
custom_scal_files = {}

# -----------------------------------------------------------------------------
# 1. GRID INGESTION & PROPERTY MAPPING
# -----------------------------------------------------------------------------
st.sidebar.subheader("1. Model Grid & Property Mapping")
uploaded_grid = st.sidebar.file_uploader(
    "Upload 3D Grid File (.GRDECL / .DATA)",
    type=["grdecl", "data", "txt", "inc"],
    help="File grid statik 3D yang memuat properti: Rock Type, Porositas, Kedalaman (Depth), dan Static Sw (SWAT)."
)

# Property keyword mapping
with st.sidebar.expander("⚙️ Konfigurasi Keyword Properti Grid", expanded=False):
    kw_rt = st.text_input("Rock Type Keyword", value="RT_HM", help="Nama keyword Rock Type pada grid (contoh: RT_HM, SATNUM, ROCKTYPE)")
    kw_sw = st.text_input("Static Sw Keyword", value="SWAT", help="Nama keyword Static Sw pada grid (contoh: SWAT, SW)")
    kw_poro = st.text_input("Porosity Keyword", value="PORO", help="Nama keyword Porositas pada grid (contoh: PORO, POR)")
    kw_depth = st.text_input("Depth Keyword", value="DEPTH", help="Nama keyword Kedalaman sel pada grid (contoh: DEPTH, TVD)")
    kw_ntg = st.text_input("Net-to-Gross Keyword (Optional)", value="NTG", help="Nama keyword NTG (opsional, kosongkan jika tidak digunakan)")
    kw_eql = st.text_input("Equil Region Keyword (Optional)", value="EQLNUM", help="Nama keyword Region EQUIL / Kompartemen (opsional, contoh: EQLNUM)")

    use_poro_cutoff = st.checkbox(
        "Filter by Minimum Porosity Cutoff",
        value=False,
        help="Centang jika model grid memuat sel matriks non-reservoar/shale berporositas nol (phi < cutoff) yang ingin dikecualikan dari optimasi."
    )
    if use_poro_cutoff:
        min_poro = st.number_input(
            "Minimum Porosity Pay Cutoff (frac)",
            value=0.001,
            min_value=0.0,
            max_value=0.5,
            step=0.001,
            format="%.4f",
            help="Sel dengan porositas di bawah nilai ini tidak akan dihitung dalam optimasi."
        )
    else:
        min_poro = 0.0

scan_res = None
if uploaded_grid:
    grid_filepath = os.path.join(temp_dir, uploaded_grid.name)
    with open(grid_filepath, "wb") as f:
        f.write(uploaded_grid.getbuffer())
    st.sidebar.caption(f"✓ Grid: `{uploaded_grid.name}` ({len(uploaded_grid.getbuffer())/1024/1024:.1f} MB)")

    # Reset session state if a different grid file is uploaded
    current_sig = f"{uploaded_grid.name}_{len(uploaded_grid.getbuffer())}_{kw_rt}_{min_poro}"
    if st.session_state.get('last_grid_signature') != current_sig:
        st.session_state['grid_data'] = None
        st.session_state['opt_results'] = None
        st.session_state['last_grid_signature'] = current_sig

        # Scan rock types from grid
        s_data, s_err = scan_grid_rock_types(grid_filepath, kw_rt, kw_depth, kw_eql, kw_poro, min_poro)
        if s_data:
            st.session_state['grid_scan_res'] = s_data
        else:
            st.session_state['grid_scan_res'] = None
            if s_err:
                st.sidebar.error(s_err)

    scan_res = st.session_state.get('grid_scan_res')


# -----------------------------------------------------------------------------
# 2. DEDICATED SCAL UPLOAD PER DETECTED ROCK TYPE
# -----------------------------------------------------------------------------
st.sidebar.subheader("2. Upload SCAL Curves per Rock Type")

if scan_res and scan_res.get('active_rts'):
    active_rts = scan_res['active_rts']
    st.sidebar.markdown(f"**Terdeteksi {len(active_rts)} Rock Type Aktif pada Grid:**")

    for r_id in active_rts:
        d_info = scan_res['rt_info'].get(r_id, {})
        z_list = d_info.get('zones', [])
        z_str = f"Zona EQL: {z_list}" if z_list else "Single Zone"
        tvd_min, tvd_max = d_info.get('tvd_min', 0.0), d_info.get('tvd_max', 0.0)
        tvd_str = f"TVD: {tvd_min:.0f}-{tvd_max:.0f}m" if tvd_max > 0 else ""
        n_pay = d_info.get('pay_cells', 0)

        st.sidebar.caption(f"📌 **Rock Type {r_id}** ({z_str} | {tvd_str} | {n_pay:,} pay cells)")
        up_rt = st.sidebar.file_uploader(
            f"Kurva SCAL untuk RT-{r_id}",
            type=["grdecl", "inc", "txt", "data"],
            key=f"scal_uploader_rt_{r_id}",
            help=f"Unggah tabel SWOF / kurva kapiler khusus untuk Rock Type {r_id} ({z_str})"
        )
        if up_rt:
            rt_save_path = os.path.join(temp_dir, f"RT_{r_id}_{up_rt.name}")
            with open(rt_save_path, "wb") as f:
                f.write(up_rt.getbuffer())
            custom_scal_files[r_id] = rt_save_path
            st.sidebar.caption(f"✓ RT-{r_id} terpasang: `{up_rt.name}`")

    # Inactive / non-reservoir rock types (if any)
    matrix_rts = [r for r in scan_res['all_rts'] if r not in active_rts]
    if matrix_rts:
        with st.sidebar.expander(f"Batuan Matriks Non-Reservoar ({matrix_rts})"):
            st.caption("Rock type berikut tidak memiliki sel pay produktif (porositas nol/tight). Opsional:")
            for r_id in matrix_rts:
                up_m = st.file_uploader(f"SCAL RT-{r_id} (Non-Pay)", type=["grdecl", "inc", "txt"], key=f"scal_uploader_rt_{r_id}")
                if up_m:
                    rt_save_path = os.path.join(temp_dir, f"RT_{r_id}_{up_m.name}")
                    with open(rt_save_path, "wb") as f:
                        f.write(up_m.getbuffer())
                    custom_scal_files[r_id] = rt_save_path

    # Bulk master file option
    with st.sidebar.expander("📁 Atau Upload 1 File Master (PROPS.INC)"):
        up_bulk = st.file_uploader(
            "Upload File Gabungan PROPS.INC",
            type=["inc", "grdecl", "data", "txt"],
            key="bulk_props_uploader",
            help="Gunakan opsi ini jika seluruh tabel SWOF untuk semua rock type digabung dalam 1 file PROPS.INC"
        )
        if up_bulk:
            bulk_path = os.path.join(temp_dir, up_bulk.name)
            with open(bulk_path, "wb") as f:
                f.write(up_bulk.getbuffer())
            props_filepath = bulk_path
            st.caption(f"✓ Master PROPS: `{up_bulk.name}`")

else:
    st.sidebar.info("ℹ️ Unggah File Grid 3D terlebih dahulu. Sistem akan otomatis mendeteksi Rock Type dan zona yang ada pada model Anda, lalu menyediakan tombol upload kurva SCAL untuk masing-masing Rock Type yang sesuai.")
    with st.sidebar.expander("📁 Upload File SCAL Manual / PROPS.INC"):
        up_manual = st.file_uploader(
            "Upload SCAL Curves (.GRDECL or .INC)",
            type=["grdecl", "inc", "txt", "data"],
            accept_multiple_files=True,
            key="manual_scal_uploader"
        )
        if up_manual:
            import re
            for uf in up_manual:
                fpath = os.path.join(temp_dir, uf.name)
                with open(fpath, "wb") as f:
                    f.write(uf.getbuffer())
                if "props" in uf.name.lower():
                    props_filepath = fpath
                else:
                    m = re.search(r'rt[-_]?(\d+)', uf.name, re.IGNORECASE)
                    rt_id = int(m.group(1)) if m else len(custom_scal_files) + 1
                    custom_scal_files[rt_id] = fpath


# -----------------------------------------------------------------------------
# 3. FLUID DECKS & UNIT CONVENTION (OPTIONAL)
# -----------------------------------------------------------------------------
st.sidebar.subheader("3. Fluid Decks & Unit Convention")
unit_mode = st.sidebar.selectbox(
    "Unit System Convention:",
    [
        "🤖 Auto-Detect (Berdasarkan File & Densitas)",
        "🌐 METRIC (Depth: m, Pc: bar, Density: kg/m³)",
        "🇺🇸 FIELD (Depth: ft, Pc: psi, Density: lb/ft³)"
    ],
    help="Sistem satuan yang digunakan model. Auto-Detect akan otomatis mendeteksi apakah model menggunakan satuan METRIC atau FIELD berdasarkan keyword FILEUNIT di SCAL dan rentang nilai DENSITY fluida."
)

uploaded_sol = st.sidebar.file_uploader(
    "Upload Solution / EQUIL (.INC / .DATA)",
    type=["inc", "data", "txt"],
    help="File solution tNavigator/ECLIPSE yang memuat keyword EQUIL (WOC/FWL, datum depth & pressure). Jika kosong, WOC diestimasi otomatis."
)
if uploaded_sol:
    sol_filepath = os.path.join(temp_dir, uploaded_sol.name)
    with open(sol_filepath, "wb") as f:
        f.write(uploaded_sol.getbuffer())
    st.sidebar.caption(f"✓ Solution: `{uploaded_sol.name}`")

uploaded_props = st.sidebar.file_uploader(
    "Upload Props / Density (.INC / .DATA)",
    type=["inc", "data", "txt"],
    help="File PROPS yang memuat keyword DENSITY (densitas fluida air, minyak, gas). Jika kosong, nilai densitas default akan digunakan."
)
if uploaded_props:
    props_filepath = os.path.join(temp_dir, uploaded_props.name)
    with open(props_filepath, "wb") as f:
        f.write(uploaded_props.getbuffer())
    st.sidebar.caption(f"✓ Props: `{uploaded_props.name}`")


# -----------------------------------------------------------------------------
# 4. OPTIMIZATION CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.subheader("4. Optimization Controls")
opt_stage = st.sidebar.selectbox(
    "Optimization Formulations:",
    [
        "Stage 3: Joint Multiplier + Endpoints + Curvature (Beta) [Recommended]",
        "Stage 2: Joint Multiplier + Endpoints (Alpha, Swirr, Pce)",
        "Stage 1: Multiplier-Only (Alpha)"
    ]
)

opt_algorithm = st.sidebar.selectbox(
    "Optimization Algorithm:",
    [
        "🧬 Differential Evolution (Global Metaheuristic) [Recommended - Best Match]",
        "🔥 Dual Annealing (Global Simulated Annealing)",
        "⚡ L-BFGS-B (Local Gradient-Based) [Fast < 1 s]"
    ],
    help="Differential Evolution mencari solusi optimal global di seluruh ruang parameter tanpa terjebak di local minima, mencegah keruntuhan kurva SCAL. L-BFGS-B lebih cepat (< 1 detik) namun rentan terjebak di lembah lokal."
)

asymp_guard = st.sidebar.radio(
    "Petrophysical Asymptotic Guard:",
    [
        "🛡️ Active (Anchor Swirr within ±20% of SCAL Lab & Preserve Asymptote) [Recommended]",
        "⚠️ Disabled (Full Unconstrained Search)"
    ],
    help="Pelindung Petrofisika: Mengunci Swirr dalam koridor ±20% dari pengukuran core laboratorium dan menjamin faktor pengali alpha mempertahankan lonjakan vertikal asimtot SCAL asli (mencegah fenomena Asymptotic Collapse seperti pada RT-4)."
)

bounds_mode = st.sidebar.radio(
    "Parameter Bounds Mode:",
    [
        "🤖 Auto-Recommended by System [Best Matching]",
        "🛠️ Manual Custom Bounds"
    ],
    help="Pilih 'Auto-Recommended' agar sistem menentukan rentang pencarian Alpha dan Beta secara optimal dan adaptif berdasarkan data reservoar untuk mencapai error minimum (RMSE terkecil)."
)

if bounds_mode.startswith("🤖"):
    alpha_max_val = 25.0
    beta_max_val = 5.0
    st.sidebar.info(
        "💡 **Rekomendasi Sistem (Aktif)**:\n"
        "- **Batas Alpha**: Adaptif dinamis [0.01 - 25.0+]\n"
        "- **Batas Beta**: Rentang Corey [0.20 - 5.00]\n"
        "- **Batas Pce**: [0.00 - 2.00 bar]\n\n"
        "*Memberikan ruang pencarian (headroom) bebas konstrain batas untuk memastikan RMSE global minimum.*"
    )
else:
    col_b1, col_b2 = st.sidebar.columns(2)
    with col_b1:
        alpha_max_val = st.number_input("Max Alpha", value=20.0, min_value=1.0, max_value=100.0, step=5.0)
    with col_b2:
        beta_max_val = st.number_input("Max Beta", value=5.0, min_value=1.0, max_value=10.0, step=1.0)


# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">⚡ Automated Static-Dynamic Sw Matching & Pc-Sw Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated spatial saturation matching between Petrel static geological models and tNavigator / ECLIPSE dynamic equilibration models with multi-region compartments (<code>EQLNUM</code>).</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "📊 1. Grid & SCAL Inspection",
    "⚙️ 2. Automated Optimization",
    "📈 3. Diagnostic Visualizations",
    "💾 4. Export Simulation Decks (.GRDECL)"
])


# -----------------------------------------------------------------------------
# TAB 1: DATA INGESTION & GRID INSPECTION
# -----------------------------------------------------------------------------
with tabs[0]:
    if grid_filepath is None or not os.path.exists(grid_filepath):
        st.info("👈 Silakan unggah File Grid (.GRDECL / .DATA) dan Kurva SCAL pada menu sidebar di sebelah kiri untuk memulai.")
        st.markdown(r"""
        ### 📋 Panduan Penggunaan Aplikasi:
        1. **1. Unggah File Grid 3D (`.GRDECL` / `.DATA`)**: Model statik hasil ekspor dari Petrel atau simulasi tNavigator/ECLIPSE yang memuat properti:
           - **Rock Type** (default keyword: `RT_HM` atau `SATNUM`)
           - **Static Water Saturation** (default keyword: `SWAT` atau `SW`)
           - **Porosity** (default keyword: `PORO` atau `POR`)
           - **Depth** (default keyword: `DEPTH` atau `TVD`)
           - *(Opsional)* **EQLNUM** (Region/Kompartemen kesetimbangan hidrostatik) dan **NTG**.
        2. **2. Unggah Kurva SCAL (`RT-*.GRDECL` atau `PROPS.INC`)**: Kurva saturasi batuan berupa file terpisah (`RT-1.GRDECL`, `RT-2.GRDECL`, ...) atau satu file gabungan `PROPS.INC` yang memuat tabel `SWOF`.
        3. **3. Solution & Fluid Decks (Opsional)**: File `.INC` atau `.DATA` yang memuat keyword `EQUIL` (kedalaman WOC & datum per region) dan `DENSITY` (densitas minyak & air). Jika tidak diunggah, level WOC akan diestimasi dari dasar reservoar dan densitas standar akan diterapkan.
        4. **4. Grid Property Mapping**: Sesuaikan nama kata kunci jika model Anda menggunakan nama properti yang berbeda.
        5. **5. Minimum Porosity Cutoff**: Centang filter jika ingin mengecualikan sel matriks non-reservoar berporositas nol ($\phi < cutoff$) dari perhitungan optimasi.
        """)
    else:
        if scan_res and scan_res.get('all_rts'):
            st.subheader("🔬 Rock Type & Zone Inventory (Hasil Scan Grid 3D)")
            st.caption("Berikut adalah ringkasan sebaran Rock Type dan zona yang terdeteksi di dalam file grid. Sistem mewajibkan kurva SCAL untuk seluruh Rock Type aktif.")

            rt_table_rows = []
            for r_id in scan_res['all_rts']:
                d = scan_res['rt_info'][r_id]
                z_str = ", ".join(map(str, d['zones'])) if d['zones'] else "Single Region"
                tvd_str = f"{d['tvd_min']:.1f} - {d['tvd_max']:.1f} m" if d['tvd_max'] > 0 else "-"

                # Check SCAL status
                if r_id in custom_scal_files:
                    fname = os.path.basename(custom_scal_files[r_id])
                    status_scal = f"✅ Terpasang (`{fname}`)"
                elif props_filepath:
                    status_scal = "✅ Terpasang via Master PROPS"
                elif not d['is_active']:
                    status_scal = "⚪ Non-Pay Matrix (Opsional)"
                else:
                    status_scal = "⚠️ Belum Diunggah (Wajib)"

                rt_table_rows.append({
                    "Rock Type": f"RT-{r_id}",
                    "Status Reservoar": "Active Pay" if d['is_active'] else "Matrix / Non-Pay",
                    "Pay Cells": f"{d['pay_cells']:,}",
                    "Depth Range TVD": tvd_str,
                    "Zona / Fault Block (EQLNUM)": z_str,
                    "Status Kurva SCAL": status_scal
                })

            st.dataframe(pd.DataFrame(rt_table_rows), use_container_width=True)

        load_col1, load_col2 = st.columns([1, 4])
        with load_col1:
            btn_load = st.button("🔄 Ingest Grid & SCAL Data", type="primary", use_container_width=True)

        # Check if missing any active rock type
        if btn_load or st.session_state['grid_data'] is None:
            missing_active = []
            if scan_res:
                for r_id in scan_res['active_rts']:
                    if r_id not in custom_scal_files and not props_filepath:
                        missing_active.append(f"RT-{r_id}")
            if missing_active:
                st.warning(f"⚠️ Kurva SCAL untuk {', '.join(missing_active)} belum diunggah. Silakan unggah kurva SCAL pada menu sidebar di sebelah kiri untuk melanjutkan.")
                st.stop()

            with st.spinner("Parsing Corner Point Grid properties and SCAL functions..."):
                t0 = time.time()
                
                # Check available keywords
                available_kw = [kw for _, kw in pcm.FastGRDECLParser.scan_keywords(grid_filepath)]
                
                # Validate required keywords
                missing_kw = []
                for label, kw_name in [("Rock Type", kw_rt), ("Porosity", kw_poro), ("Depth", kw_depth), ("Static Sw", kw_sw)]:
                    if kw_name not in available_kw:
                        missing_kw.append(f"{label} (`{kw_name}`)")
                
                if missing_kw:
                    st.error(f"❌ Kata kunci berikut tidak ditemukan di dalam file grid: {', '.join(missing_kw)}.\n\n"
                             f"Kata kunci yang terdeteksi di dalam file Anda: `{', '.join(available_kw)}`.\n\n"
                             f"Silakan sesuaikan nama kata kunci pada menu sidebar **1. Model Grid & Property Mapping**.")
                    st.stop()

                # Parse properties
                rt_grid = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_rt)
                tot_cells = len(rt_grid)
                poro_grid = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_poro, total_cells=tot_cells)
                depth_grid = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_depth, total_cells=tot_cells)
                sw_static_grid = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_sw, total_cells=tot_cells)
                
                ntg_grid = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_ntg, total_cells=tot_cells) if (kw_ntg and kw_ntg in available_kw) else None
                eqlnum_grid = pcm.FastGRDECLParser.parse_property(grid_filepath, kw_eql, total_cells=tot_cells, dtype=np.int32) if (kw_eql and kw_eql in available_kw) else None

                # Extract active reservoir pay
                active_mask, active_stats = pcm.extract_active_reservoir(rt_grid, poro_grid, ntg_grid, min_poro=min_poro)
                
                if np.sum(active_mask) == 0:
                    st.warning("⚠️ Tidak ada sel reservoar aktif yang memenuhi kriteria (Rock Type > 0 dan Porositas >= cutoff). Periksa kembali nilai cut-off atau keyword Rock Type.")
                    st.stop()

                # Load SCAL Manager
                scal_mgr = pcm.SCALManager(temp_dir)
                if custom_scal_files:
                    for rt_id, sc_path in sorted(custom_scal_files.items()):
                        scal_mgr.load_rock_type(rt_id, filename=sc_path)
                elif props_filepath and os.path.exists(props_filepath):
                    scal_mgr.load_from_props_file(props_filepath)

                if len(scal_mgr.tables) == 0:
                    st.error("⚠️ Kurva SCAL Rock Type belum diunggah atau tidak ditemukan tabel SWOF. Silakan unggah file `RT-*.GRDECL` atau `PROPS.INC` pada menu sidebar.")
                    st.stop()

                # Parse Solution / Equil & Density
                multi_params = {}
                density_records = pcm.FastGRDECLParser.parse_density_keyword(props_filepath) if props_filepath and os.path.exists(props_filepath) else []
                equil_records = pcm.FastGRDECLParser.parse_equil_keyword(sol_filepath) if sol_filepath and os.path.exists(sol_filepath) else []

                rho_w = density_records[0]['water_density'] if density_records else 1013.2
                rho_o = density_records[1]['oil_density'] if len(density_records) > 1 else (density_records[0]['oil_density'] if density_records else 820.91)

                # Auto-detect or user-selected unit system
                detected_unit = "METRIC"
                if "METRIC" in unit_mode:
                    detected_unit = "METRIC"
                elif "FIELD" in unit_mode:
                    detected_unit = "FIELD"
                else:  # Auto-detect
                    scal_units = [tbl.fileunit.upper() for tbl in scal_mgr.tables.values()] if scal_mgr.tables else []
                    if "FIELD" in scal_units:
                        detected_unit = "FIELD"
                    elif rho_w < 200.0:  # Field water density is ~62-65 lb/ft^3; METRIC is ~1000 kg/m^3
                        detected_unit = "FIELD"
                    else:
                        detected_unit = "METRIC"

                p_unit = "psi" if detected_unit == "FIELD" else "bar"
                d_unit = "ft" if detected_unit == "FIELD" else "m"
                rho_unit = "lb/ft³" if detected_unit == "FIELD" else "kg/m³"

                # Sync table fileunit with detected unit system for export
                for tbl in scal_mgr.tables.values():
                    tbl.fileunit = detected_unit

                if equil_records:
                    for eq in equil_records:
                        r_id = eq['region']
                        multi_params[r_id] = pcm.EquilibrationParams(
                            fwl_depth=float(eq['woc_depth']),
                            datum_depth=float(eq['datum_depth']),
                            datum_pressure=float(eq['datum_pressure']),
                            water_density=float(rho_w),
                            oil_density=float(rho_o),
                            unit_system=detected_unit
                        )
                else:
                    # Single default equilibration datum
                    multi_params[1] = pcm.EquilibrationParams(
                        fwl_depth=float(np.max(depth_grid[active_mask])),
                        datum_depth=float(np.mean(depth_grid[active_mask])),
                        datum_pressure=100.0 if detected_unit == "METRIC" else 1450.0,
                        water_density=float(rho_w),
                        oil_density=float(rho_o),
                        unit_system=detected_unit
                    )

                equilibrator = pcm.ReservoirEquilibrator(scal_mgr, multi_params=multi_params)

                # Store in session state
                st.session_state['grid_data'] = {
                    'total_cells': tot_cells,
                    'active_mask': active_mask,
                    'rt_active': rt_grid[active_mask].astype(int),
                    'poro_active': poro_grid[active_mask],
                    'depth_active': depth_grid[active_mask],
                    'sw_static_active': sw_static_grid[active_mask],
                    'eqlnum_active': eqlnum_grid[active_mask] if eqlnum_grid is not None else np.ones(np.sum(active_mask), dtype=int),
                    'density_records': density_records,
                    'equil_records': equil_records,
                    'multi_params': multi_params,
                    'unit_system': detected_unit,
                    'p_unit': p_unit,
                    'd_unit': d_unit,
                    'rho_unit': rho_unit
                }
                st.session_state['scal_manager'] = scal_mgr
                st.session_state['equilibrator'] = equilibrator
                st.success(f"Ingested {tot_cells:,} grid cells and {len(scal_mgr.tables)} rock types in {time.time()-t0:.2f} s!")

        if st.session_state['grid_data']:
            gd = st.session_state['grid_data']
            sm = st.session_state['scal_manager']

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Grid Cells</div>
                    <div class="metric-value">{gd['total_cells']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Active Reservoir Pay Cells</div>
                    <div class="metric-value">{len(gd['depth_active']):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Loaded SCAL Rock Types</div>
                    <div class="metric-value">{len(sm.tables)} Rock Types</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Equilibration Compartments</div>
                    <div class="metric-value">{len(gd['multi_params'])} Regions</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")

            # Active rock types summary table
            p_unit = gd.get('p_unit', 'bar')
            d_unit = gd.get('d_unit', 'm')
            st.subheader(f"Rock Type Pay Zone Distribution (Unit System: {gd.get('unit_system', 'METRIC')})")
            rt_summary = []
            for r in np.unique(gd['rt_active']):
                m = (gd['rt_active'] == r)
                rt_summary.append({
                    "Rock Type": f"RT-{r}",
                    "Active Cells": f"{np.sum(m):,}",
                    "Pay Fraction": f"{np.sum(m)/len(gd['rt_active'])*100:.2f}%",
                    "Mean TVD": f"{gd['depth_active'][m].mean():.1f} {d_unit}",
                    "Mean Porosity": f"{gd['poro_active'][m].mean():.3f}",
                    "Mean Static Sw": f"{gd['sw_static_active'][m].mean():.3f}",
                    "SCAL Base Sw_irr": f"{sm.tables[r].base_swirr:.3f}" if r in sm.tables else "-"
                })
            st.dataframe(pd.DataFrame(rt_summary), use_container_width=True)

            # Fault Compartments / EQLNUM summary
            if len(gd['multi_params']) > 1:
                st.subheader("Equilibration Compartments (EQLNUM)")
                eq_rows = []
                for r_id, p in gd['multi_params'].items():
                    m = (gd['eqlnum_active'] == r_id)
                    eq_rows.append({
                        "Region": f"EQLNUM {r_id}",
                        f"WOC / FWL Depth [{d_unit} TVD]": f"{p.fwl_depth:.2f}",
                        f"Capillary Gradient [{p_unit}/{d_unit}]": f"{p.capillary_gradient:.6f}",
                        "Active Cells": f"{np.sum(m):,}",
                        "Mean Sw Static": f"{gd['sw_static_active'][m].mean():.3f}" if np.any(m) else "-"
                    })
                st.dataframe(pd.DataFrame(eq_rows), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: AUTOMATED OPTIMIZATION & CONVERGENCE
# -----------------------------------------------------------------------------
with tabs[1]:
    if st.session_state['grid_data'] is None:
        st.warning("Please ingest data in Tab 1 first.")
    else:
        gd = st.session_state['grid_data']
        sm = st.session_state['scal_manager']
        eq = st.session_state['equilibrator']

        st.subheader("Capillary Pressure Optimization Execution")
        st.write("The optimizer minimizes spatial saturation mismatch between Petrel static Sw and tNavigator hydrostatic equilibrium Sw.")

        opt_domain = st.radio(
            "Optimization Domain:",
            ["Hydrocarbon Column Only (Transition Zone, Pc > 0) [Physically Consistent]", "All Active Cells"],
            index=0,
            horizontal=True,
            help="Capillary pressure curves only govern saturation in the hydrocarbon column (Pc > 0). Below WOC, thermodynamics fixes Sw = 1.0."
        )

        btn_run_opt = st.button("🚀 Run Automated Optimization", type="primary", use_container_width=True)

        if btn_run_opt or st.session_state['opt_results'] is not None:
            if btn_run_opt:
                with st.spinner("Solving multi-stage optimization across 3D cells..."):
                    t_start = time.time()
                    
                    # Compute hydrostatic Pc
                    pc_hydro = eq.compute_hydrostatic_pc(gd['depth_active'], eqlnum_array=gd['eqlnum_active'])
                    oil_mask = (pc_hydro > 0) if "Hydrocarbon" in opt_domain else np.ones(len(gd['depth_active']), dtype=bool)

                    # Determine effective optimization bounds
                    if bounds_mode.startswith("🤖"):
                        max_h_pc = float(np.max(pc_hydro[oil_mask])) if np.any(oil_mask) else 10.0
                        max_scal_pc = max([tbl.base_pcow.max() for tbl in sm.tables.values()]) if sm.tables else 1.0
                        adaptive_alpha = max(25.0, min(60.0, float(np.ceil((max_h_pc / max_scal_pc) * 2.5)))) if max_scal_pc > 0 else 25.0
                        eff_alpha_max = adaptive_alpha
                        eff_beta_max = 5.0
                    else:
                        eff_alpha_max = float(alpha_max_val)
                        eff_beta_max = float(beta_max_val)

                    # Optimizer instance
                    optimizer = pcm.PcOptimizer(
                        equilibrator=eq,
                        depth_active=gd['depth_active'][oil_mask],
                        rt_active=gd['rt_active'][oil_mask],
                        sw_static_active=gd['sw_static_active'][oil_mask],
                        eqlnum_active=gd['eqlnum_active'][oil_mask]
                    )

                    # Stage 1: Multiplier-only
                    res_stage1 = optimizer.optimize_independent(bounds=(0.01, eff_alpha_max))
                    
                    # Stage 2 / 3: Multiplier + Endpoints + Curvature
                    use_curv = "Curvature (Beta)" in opt_stage
                    use_pce = "Endpoints" in opt_stage or use_curv
                    eff_pce_max = 30.0 if gd.get('unit_system') == 'FIELD' else 2.0

                    # Parse solver algorithm choice
                    if "Differential" in opt_algorithm:
                        algo_choice = "differential_evolution"
                    elif "Annealing" in opt_algorithm:
                        algo_choice = "dual_annealing"
                    else:
                        algo_choice = "L-BFGS-B"

                    use_guard = asymp_guard.startswith("🛡️")

                    res_stage2 = optimizer.optimize_endpoints(
                        alpha_bounds=(0.01, eff_alpha_max),
                        pce_bounds=(0.0, eff_pce_max),
                        beta_bounds=(0.2, eff_beta_max),
                        optimize_entry_pressure=use_pce,
                        optimize_curvature=use_curv,
                        algorithm=algo_choice,
                        asymptotic_guard=use_guard
                    )

                    # Generate dynamic Sw across all active cells
                    sw_dyn_init = eq.calculate_dynamic_sw(gd['depth_active'], gd['rt_active'], {r: 1.0 for r in sm.tables}, eqlnum_array=gd['eqlnum_active'])
                    sw_dyn_mult = eq.calculate_dynamic_sw(gd['depth_active'], gd['rt_active'], res_stage1['optimal_multipliers'], eqlnum_array=gd['eqlnum_active'])
                    sw_dyn_opt = eq.calculate_dynamic_sw(gd['depth_active'], gd['rt_active'], res_stage2['optimal_parameters'], eqlnum_array=gd['eqlnum_active'])

                    # Compute metrics
                    m_init = pcm.OptimizationMetrics.compute(gd['sw_static_active'][oil_mask], sw_dyn_init[oil_mask], gd['rt_active'][oil_mask])
                    m_stg1 = res_stage1['optimal_metrics']
                    m_stg2 = res_stage2['optimal_metrics']

                    st.session_state['opt_results'] = {
                        'stage1': res_stage1,
                        'stage2': res_stage2,
                        'init_metrics': m_init,
                        'stg1_metrics': m_stg1,
                        'stg2_metrics': m_stg2,
                        'oil_mask': oil_mask,
                        'sw_dyn_init': sw_dyn_init,
                        'sw_dyn_mult': sw_dyn_mult,
                        'sw_dyn_opt': sw_dyn_opt,
                        'solve_time': time.time() - t_start,
                        'eff_alpha_max': eff_alpha_max,
                        'eff_beta_max': eff_beta_max,
                        'bounds_mode': bounds_mode,
                        'algorithm': algo_choice,
                        'asymptotic_guard': use_guard
                    }
                    st.success(f"Optimization converged in {st.session_state['opt_results']['solve_time']:.2f} seconds!")

            # Display results
            if st.session_state['opt_results']:
                res = st.session_state['opt_results']
                m_init = res['init_metrics']
                m_opt = res['stg2_metrics']

                if res.get('asymptotic_guard'):
                    st.success(r"🛡️ **Petrophysical Asymptotic Guard Active**: Seluruh kurva SCAL diikat ke nilai inti batuan laboratorium ($S_{wirr} \pm 20\%$ & $\alpha \ge 0.80$). Dinding asimtot vertikal terproteksi penuh dari fenomena *Asymptotic Collapse*.")

                algo_name = "🧬 Differential Evolution (Global)" if res.get('algorithm') == 'differential_evolution' else ("🔥 Dual Annealing (Global)" if res.get('algorithm') == 'dual_annealing' else "⚡ L-BFGS-B (Local)")
                st.info(f"⚙️ **Optimization Engine**: `{algo_name}` | Bounds Mode: `{res.get('bounds_mode', 'Auto')}` | Solved across {len(res['stage2']['per_rock_type'])} Rock Types in {res.get('solve_time', 0):.2f} seconds.")

                if res.get('bounds_mode', '').startswith("🤖"):
                    st.caption(f"🤖 **Auto-Recommended Formulation Applied**: Batas optimasi adaptif otomatis diatur ke Alpha ∈ [0.01, {res.get('eff_alpha_max', 25.0):.1f}] dan Beta ∈ [0.20, {res.get('eff_beta_max', 5.0):.1f}] untuk memastikan konvergensi global dan residual error minimum.")

                err_red = (1.0 - m_opt['global_rmse'] / m_init['global_rmse']) * 100.0 if m_init['global_rmse'] > 0 else 0.0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Initial Baseline RMSE", f"{m_init['global_rmse']:.4f}")
                with c2:
                    st.metric("Optimized RMSE", f"{m_opt['global_rmse']:.4f}", delta=f"-{err_red:.1f}% error", delta_color="inverse")
                with c3:
                    st.metric("Parity R² Score", f"{m_opt['r2_score']:.4f}", delta=f"{m_opt['r2_score']-m_init['r2_score']:+.4f}")
                with c4:
                    st.metric("Residual Bias", f"{m_opt['mean_bias']:+.5f}")

                st.write("---")
                st.subheader("Rock Type Optimal Parameter Breakdown")
                
                table_records = []
                p_unit = gd.get('p_unit', 'bar')
                for r in sorted(res['stage2']['per_rock_type'].keys()):
                    info1 = res['stage1']['per_rock_type'][r]
                    info2 = res['stage2']['per_rock_type'][r]
                    red_pct = (1.0 - info2['optimal_rmse'] / info1['initial_rmse']) * 100.0 if info1['initial_rmse'] > 0 else 0.0
                    table_records.append({
                        "Rock Type": f"RT-{r}",
                        "Base Sw_irr": f"{info2['base_swirr']:.3f}",
                        "Optimal Sw_irr": f"{info2['optimal_swirr']:.4f}",
                        "Stage 1 Alpha*": f"{info1['optimal_multiplier']:.3f}",
                        "Optimal Alpha*": f"{info2['optimal_alpha']:.3f}",
                        f"Optimal Pce [{p_unit}]": f"{info2['optimal_pce']:.4f}",
                        "Optimal Beta": f"{info2['optimal_beta']:.4f}",
                        "Baseline RMSE": f"{info1['initial_rmse']:.4f}",
                        "Optimized RMSE": f"{info2['optimal_rmse']:.4f}",
                        "Error Reduction": f"{red_pct:.1f}%"
                    })
                st.dataframe(pd.DataFrame(table_records), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: DIAGNOSTIC VISUALIZATIONS
# -----------------------------------------------------------------------------
with tabs[2]:
    if st.session_state['opt_results'] is None:
        st.info("Run optimization in Tab 2 to view diagnostic plots.")
    else:
        res = st.session_state['opt_results']
        gd = st.session_state['grid_data']
        sm = st.session_state['scal_manager']
        oil_mask = res['oil_mask']

        diag_choice = st.selectbox(
            "Select Diagnostic Figure:",
            [
                "1. SCAL Capillary Pressure Curves (Pc vs Sw)",
                "2. Saturation vs. True Vertical Depth (Sw vs TVD)",
                "3. Parity Cross-Plots Progression (Static vs Dynamic)",
                "4. Residual Error Distribution Analysis"
            ]
        )

        if "1. SCAL" in diag_choice:
            st.subheader("SCAL Capillary Pressure Curve Progression")
            active_rts = sorted(list(sm.tables.keys()))
            fig = make_subplots(rows=1, cols=len(active_rts), subplot_titles=[f"Rock Type {r}" for r in active_rts])

            for idx, r in enumerate(active_rts, start=1):
                tbl = sm.tables[r]
                base_sw = tbl.base_sw
                base_pc = tbl.base_pcow
                s_wn = (base_sw - tbl.base_swirr) / (1.0 - tbl.base_swirr)

                # Base curve
                fig.add_trace(go.Scatter(x=base_sw, y=base_pc, mode='lines', name=f'Base RT-{r}', line=dict(color='black', dash='dash')), row=1, col=idx)

                # Stage 1 Multiplier curve
                mult = res['stage1']['optimal_multipliers'].get(r, 1.0)
                fig.add_trace(go.Scatter(x=base_sw, y=base_pc * mult, mode='lines', name=f'Stage 1 (α={mult:.2f})', line=dict(color='blue', dash='dot')), row=1, col=idx)

                # Stage 2 Optimized curve
                p = res['stage2']['optimal_parameters'].get(r, {})
                opt_a = p.get('alpha', 1.0)
                opt_sw = p.get('swirr', tbl.base_swirr)
                opt_pce = p.get('pce', 0.0)
                opt_beta = p.get('beta', 1.0)

                s_wn_curv = s_wn ** opt_beta
                sw_pts = opt_sw + s_wn_curv * (1.0 - opt_sw)
                pc_pts = base_pc * opt_a + opt_pce * (1.0 - s_wn_curv)
                pc_pts[-1] = 0.0

                lbl = f'Opt (α={opt_a:.2f}, Swirr={opt_sw:.2f}, β={opt_beta:.2f})'
                fig.add_trace(go.Scatter(x=sw_pts, y=pc_pts, mode='lines+markers', name=lbl, line=dict(color='red', width=3)), row=1, col=idx)

                fig.update_xaxes(title_text="Water Saturation Sw [frac]", range=[0, 1.05], row=1, col=idx)
                fig.update_yaxes(title_text=f"Capillary Pressure Pcow [{gd.get('p_unit', 'bar')}]" if idx == 1 else None, row=1, col=idx)

            fig.update_layout(height=480, template="plotly_white", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        elif "2. Saturation vs. True Vertical Depth" in diag_choice:
            st.subheader("Saturation vs. Depth Distribution across Pay Facies")
            unique_rts = sorted(list(np.unique(gd['rt_active'][oil_mask])))
            fig = make_subplots(rows=1, cols=len(unique_rts), subplot_titles=[f"Rock Type {r}" for r in unique_rts], shared_yaxes=True)

            # Downsample for snappy Plotly rendering
            sample_n = min(3000, np.sum(oil_mask))
            idx_sample = np.random.choice(np.where(oil_mask)[0], sample_n, replace=False)

            d_s = gd['depth_active'][idx_sample]
            rt_s = gd['rt_active'][idx_sample]
            sw_stat_s = gd['sw_static_active'][idx_sample]
            sw_dyn_s = res['sw_dyn_opt'][idx_sample]

            for col_idx, r in enumerate(unique_rts, start=1):
                m_r = (rt_s == r)
                fig.add_trace(go.Scatter(x=sw_stat_s[m_r], y=d_s[m_r], mode='markers', name=f'Static Sw RT-{r}', marker=dict(size=4, color='blue', opacity=0.35)), row=1, col=col_idx)
                fig.add_trace(go.Scatter(x=sw_dyn_s[m_r], y=d_s[m_r], mode='markers', name=f'Optimized Dynamic RT-{r}', marker=dict(size=4, color='red', opacity=0.55)), row=1, col=col_idx)
                fig.update_xaxes(title_text="Water Saturation Sw", range=[0, 1.05], row=1, col=col_idx)
                if col_idx == 1:
                    fig.update_yaxes(title_text=f"Depth TVD [{gd.get('d_unit', 'm')}]", autorange="reversed", row=1, col=col_idx)
                else:
                    fig.update_yaxes(autorange="reversed", row=1, col=col_idx)

            fig.update_layout(height=520, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        elif "3. Parity Cross-Plots" in diag_choice:
            st.subheader("Parity Cross-Plots: Static Sw vs Dynamic Sw Progression")
            fig = make_subplots(rows=1, cols=3, subplot_titles=["Baseline (Uncalibrated)", "Stage 1 (Multiplier-Only)", "Stage 2 (Optimized)"])

            sample_n = min(4000, np.sum(oil_mask))
            idx_sample = np.random.choice(np.where(oil_mask)[0], sample_n, replace=False)

            y_stat = gd['sw_static_active'][idx_sample]
            x_init = res['sw_dyn_init'][idx_sample]
            x_stg1 = res['sw_dyn_mult'][idx_sample]
            x_opt = res['sw_dyn_opt'][idx_sample]

            # 1:1 Reference lines
            for col_idx in [1, 2, 3]:
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', dash='dash'), name='1:1 Parity'), row=1, col=col_idx)

            fig.add_trace(go.Scatter(x=x_init, y=y_stat, mode='markers', marker=dict(size=3, color='black', opacity=0.3), name='Baseline'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_stg1, y=y_stat, mode='markers', marker=dict(size=3, color='blue', opacity=0.3), name='Stage 1'), row=1, col=2)
            fig.add_trace(go.Scatter(x=x_opt, y=y_stat, mode='markers', marker=dict(size=3, color='red', opacity=0.4), name='Optimized'), row=1, col=3)

            for col_idx in [1, 2, 3]:
                fig.update_xaxes(title_text="Dynamic Sw [frac]", range=[0, 1], row=1, col=col_idx)
                fig.update_yaxes(title_text="Static Sw [frac]" if col_idx == 1 else None, range=[0, 1], row=1, col=col_idx)

            fig.update_layout(height=450, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "4. Residual Error" in diag_choice:
            st.subheader("Residual Error Distribution (Dynamic Sw - Static Sw)")
            err_init = res['sw_dyn_init'][oil_mask] - gd['sw_static_active'][oil_mask]
            err_opt = res['sw_dyn_opt'][oil_mask] - gd['sw_static_active'][oil_mask]

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=err_init, name="Baseline Residual Error", opacity=0.45, marker_color="black", nbinsx=60))
            fig.add_trace(go.Histogram(x=err_opt, name="Optimized Residual Error", opacity=0.6, marker_color="red", nbinsx=60))
            fig.update_layout(barmode='overlay', xaxis_title="Saturation Error ΔSw = Sw_dyn - Sw_stat", yaxis_title="Cell Frequency", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: EXPORT SIMULATION INCLUDE FILES (.GRDECL)
# -----------------------------------------------------------------------------
with tabs[3]:
    if st.session_state['opt_results'] is None:
        st.info("Run optimization in Tab 2 to generate simulation include decks.")
    else:
        sm = st.session_state['scal_manager']
        opt_res = st.session_state['opt_results']
        stg1_params = opt_res['stage1']['optimal_multipliers']
        stg2_params = opt_res['stage2']['optimal_parameters']
        gd = st.session_state['grid_data']
        p_unit = gd.get('p_unit', 'bar')
        active_rts = sorted(list(sm.tables.keys()))

        st.subheader("Simulation-Ready Include Files (.GRDECL) Export")
        st.write("Ekspor kurva saturasi SCAL hasil optimasi ke dalam format keyword `SWOF` dan `SGOF` standar ECLIPSE & tNavigator.")

        # Pre-generate Stage 1 and Stage 2 ZIP archives for instant 1-click download
        # Stage 1 ZIP (Multiplier-Only)
        sm.apply_multipliers_dict(stg1_params)
        zip_buf_stg1 = io.BytesIO()
        with zipfile.ZipFile(zip_buf_stg1, "w", zipfile.ZIP_DEFLATED) as zf1:
            for r_id in active_rts:
                c1 = sm.export_scal_string(r_id, stage_label="Stage 1 (Multiplier-Only)")
                zf1.writestr(f"RT-{r_id}_opt.GRDECL", c1)

        # Stage 2/3 ZIP (Multi-Parameter)
        sm.apply_multipliers_dict(stg2_params)
        zip_buf_stg2 = io.BytesIO()
        with zipfile.ZipFile(zip_buf_stg2, "w", zipfile.ZIP_DEFLATED) as zf2:
            for r_id in active_rts:
                c2 = sm.export_scal_string(r_id, stage_label="Stage 2/3 (Multi-Parameter)")
                zf2.writestr(f"RT-{r_id}_opt.GRDECL", c2)

        # Quick Batch Download Buttons
        st.markdown("#### 📦 Quick Batch Download (.ZIP)")
        col_z1, col_z2 = st.columns(2)
        with col_z1:
            st.download_button(
                label="⚡ Download All as Stage 1 (.ZIP)\n(Multiplier-Only / Lab Curvature Preserved)",
                data=zip_buf_stg1.getvalue(),
                file_name="Optimized_SCAL_Decks_Stage1_MultiplierOnly.zip",
                mime="application/zip",
                use_container_width=True
            )
            st.caption("Direkomendasikan jika ingin mempertahankan kurva lab asli dan mencegah keruntuhan asimtot (misal RT-4).")

        with col_z2:
            st.download_button(
                label="🎯 Download All as Stage 2/3 (.ZIP)\n(Full Multi-Parameter: Alpha, Swirr, Pce, Beta)",
                data=zip_buf_stg2.getvalue(),
                file_name="Optimized_SCAL_Decks_Stage2_MultiParam.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            st.caption("Kalibrasi penuh seluruh parameter untuk meminimalkan spatial saturation mismatch (RMSE minimum).")

        st.write("---")

        # Interactive Formulation Selector for Preview & Customization
        st.markdown("#### 🎛️ Konfigurasi Formulasi Ekspor & Code Preview")
        export_mode = st.radio(
            "Pilih formulasi aktif untuk pratinjau tabel dan download file individual:",
            [
                "⚡ Stage 1 (Multiplier-Only): Seluruh Rock Type",
                "🎯 Stage 2/3 (Multi-Parameter): Seluruh Rock Type",
                "🔀 Kustom / Hybrid: Tentukan Formulasi per Rock Type"
            ],
            index=0 if "Stage 1" in opt_stage else 1,
            horizontal=False,
            help="Pilih Stage 1 untuk menjaga kelengkungan lab asli, Stage 2/3 untuk tuning multi-parameter, atau Hybrid untuk mencampur (misal RT-4 memakai Stage 1 dan lainnya memakai Stage 2/3)."
        )

        selected_export_params = {}
        stage_descriptions = {}

        if export_mode.startswith("⚡ Stage 1"):
            for r in active_rts:
                selected_export_params[r] = stg1_params.get(r, 1.0)
                stage_descriptions[r] = "Stage 1 (Multiplier-Only)"
            active_zip_name = "Optimized_SCAL_Decks_Stage1_MultiplierOnly.zip"
            st.info("⚡ **Mode Aktif: Stage 1 (Multiplier-Only)** — Hanya mengubah skala vertikal Pc dengan faktor pengali $\\alpha^*$. Bentuk kelengkungan asimtotik data laboratorium asli, titik $S_{wirr}$, dan $P_{ce}=0$ dipertahankan 100%.")

        elif export_mode.startswith("🎯 Stage 2/3"):
            for r in active_rts:
                selected_export_params[r] = stg2_params.get(r, {'alpha': 1.0, 'swirr': sm.tables[r].base_swirr, 'pce': 0.0, 'beta': 1.0})
                stage_descriptions[r] = "Stage 2/3 (Multi-Parameter)"
            active_zip_name = "Optimized_SCAL_Decks_Stage2_MultiParam.zip"
            st.info("🎯 **Mode Aktif: Stage 2/3 (Multi-Parameter)** — Kalibrasi komprehensif seluruh parameter ($\\alpha^*, S_{wirr}^*, P_{ce}^*, \\beta^*$). Memberikan minimasi residual RMSE global terendah.")

        else:  # Hybrid
            st.write("🔧 **Pengaturan Formulasi per Rock Type (Kustom / Hybrid)**:")
            st.caption("Sesuaikan formulasi untuk masing-masing tipe batuan. Contoh petrofisika: Gunakan **Stage 1** untuk Rock Type dengan head kapiler rendah atau asimtot yang perlu dipertahankan (seperti RT-4), dan **Stage 2/3** untuk Rock Type lainnya (RT-1, RT-2, RT-3).")
            
            h_cols = st.columns(min(len(active_rts), 4))
            for i, r in enumerate(active_rts):
                col = h_cols[i % len(h_cols)]
                with col:
                    choice = st.radio(
                        f"Formulasi RT-{r}:",
                        ["⚡ Stage 1 (Multiplier)", "🎯 Stage 2/3 (Multi-Param)"],
                        key=f"hybrid_choice_{r}",
                        index=0 if r == 4 else 1  # Default RT-4 to Stage 1
                    )
                    if "Stage 1" in choice:
                        selected_export_params[r] = stg1_params.get(r, 1.0)
                        stage_descriptions[r] = "Stage 1 (Multiplier-Only)"
                    else:
                        selected_export_params[r] = stg2_params.get(r, {'alpha': 1.0, 'swirr': sm.tables[r].base_swirr, 'pce': 0.0, 'beta': 1.0})
                        stage_descriptions[r] = "Stage 2/3 (Multi-Parameter)"

            active_zip_name = "Optimized_SCAL_Decks_Hybrid.zip"

            # Generate Hybrid ZIP
            sm.apply_multipliers_dict(selected_export_params)
            zip_buf_hybrid = io.BytesIO()
            with zipfile.ZipFile(zip_buf_hybrid, "w", zipfile.ZIP_DEFLATED) as zf_h:
                for r_id in active_rts:
                    c_h = sm.export_scal_string(r_id, stage_label=stage_descriptions[r_id])
                    zf_h.writestr(f"RT-{r_id}_opt.GRDECL", c_h)

            st.download_button(
                label="📦 Download Selected Hybrid Decks (.ZIP)",
                data=zip_buf_hybrid.getvalue(),
                file_name=active_zip_name,
                mime="application/zip",
                type="primary",
                use_container_width=True
            )

        # Apply the chosen formulation to scal manager for preview
        sm.apply_multipliers_dict(selected_export_params)

        st.write("---")
        st.subheader("Individual Rock Type Decks & Code Preview")

        rt_tabs = st.tabs([f"RT-{r} [{stage_descriptions[r].split()[0]}]" for r in active_rts])
        for idx, r_id in enumerate(active_rts):
            with rt_tabs[idx]:
                tbl = sm.tables[r_id]
                current_formulation = stage_descriptions[r_id]
                content = sm.export_scal_string(r_id, stage_label=current_formulation)

                # Parameter summary card
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Formulasi", current_formulation)
                with m2:
                    st.metric("Alpha (Multiplier)", f"{tbl.current_multiplier:.4f}")
                with m3:
                    st.metric("Sw_irr (Endpoint)", f"{tbl.current_swirr:.4f}")
                with m4:
                    st.metric(f"Pce [{p_unit}] / Beta", f"{tbl.current_pce:.2f} {p_unit} / {tbl.current_beta:.2f}")

                # Download buttons for this rock type
                b_col1, b_col2 = st.columns([1, 1])
                with b_col1:
                    st.download_button(
                        label=f"⬇️ Download RT-{r_id}_opt.GRDECL ({current_formulation})",
                        data=content,
                        file_name=f"RT-{r_id}_opt.GRDECL",
                        mime="text/plain",
                        key=f"dl_opt_{r_id}"
                    )
                with b_col2:
                    # Provide direct Stage 1 download if currently looking at Stage 2 or vice versa
                    alt_label = "Stage 1" if "Stage 2" in current_formulation else "Stage 2/3"
                    alt_param = stg1_params.get(r_id, 1.0) if "Stage 2" in current_formulation else stg2_params.get(r_id, {'alpha': 1.0, 'swirr': tbl.base_swirr, 'pce': 0.0, 'beta': 1.0})
                    
                    # Temporarily compute alt content
                    sm.apply_multipliers_dict({r_id: alt_param})
                    alt_content = sm.export_scal_string(r_id, stage_label=f"Alternative: {alt_label}")
                    sm.apply_multipliers_dict(selected_export_params)  # restore
                    
                    st.download_button(
                        label=f"⬇️ Download RT-{r_id}_{alt_label.replace('/', '_')}.GRDECL",
                        data=alt_content,
                        file_name=f"RT-{r_id}_{alt_label.replace('/', '_')}.GRDECL",
                        mime="text/plain",
                        key=f"dl_alt_{r_id}"
                    )

                st.code(content, language="text")

        st.write("---")
        st.markdown(r"""
        ### Instructions for Manual Execution in tNavigator:
        1. **Update PROPS Section**: Masukkan file `RT-*_opt.GRDECL` hasil unduhan ke dalam deck simulasi `.DATA`:
           ```text
           PROPS
           INCLUDE 'RT-1_opt.GRDECL' /
           INCLUDE 'RT-2_opt.GRDECL' /
           INCLUDE 'RT-3_opt.GRDECL' /
           INCLUDE 'RT-4_opt.GRDECL' /
           ```
        2. **Run Initialization**: Jalankan Step 0 (Equilibration / `EQUIL`) di tNavigator.
        3. **Export Dynamic Sw**: Pada viewer 3D tNavigator, pilih properti `SWAT` pada timestep 0 $\to$ Export Property untuk verifikasi kecocokan saturasi 3D.
        """)
