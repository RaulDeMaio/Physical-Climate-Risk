import streamlit as st
import sys
import os

try:
    from app.utils import setup_paths
except ImportError:
    sys.path.append(os.getcwd())
    from app.utils import setup_paths

setup_paths()

from app.data import load_core_model_data, load_calibration_data
from src.io_climate.calibration import shock_scalar
from io_climate.postprocess import postprocess_results
from io_climate.viz import build_dashboard_bundle, plot_top_countries, iso2_to_iso3
from app.branding import set_streamlit_branding, apply_oe_branding

# --- 2. Page Config ---
st.set_page_config(
    page_title="Physical Climate Risk App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 3. UI Helper Functions ---
def get_country_list(node_labels):
    return sorted(list({lbl.split("::")[0] for lbl in node_labels}))


def get_sector_list(node_labels, country):
    prefix = f"{country}::"
    return [lbl.split("::")[1] for lbl in node_labels if lbl.startswith(prefix)]


def format_sector_label(code, decoder):
    name = decoder.get(code, code)
    return f"{code} - {name[:60]}..." if len(name) > 60 else f"{code} - {name}"


# --- 4. Main Interface ---
def main():
    set_streamlit_branding()
    st.title("🌍 Physical Climate Risk Propagation [v1.4]")

    # --- Data Loading ---
    with st.spinner("Initializing Model & Data..."):
        try:
            inputs = load_core_model_data()
            pct_table = load_calibration_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    model = inputs.model_instance

    # --- Sidebar Controls ---
    st.sidebar.header("Scenario Configuration")
    mode = st.sidebar.radio("Simulation Mode", ["Hazard Calibration", "Manual Shock"])

    run_params = {}

    if mode == "Hazard Calibration":
        st.sidebar.subheader("Hazard Parameters")
        hazard_opts = sorted(pct_table["hazard"].unique())
        countries = sorted(pct_table["ISO2"].unique())
        levels = ["moderate", "severe", "extreme", "very_extreme"]

        sel_country = st.sidebar.selectbox(
            "Country",
            countries,
            index=countries.index("IT") if "IT" in countries else 0,
        )
        sel_hazard = st.sidebar.selectbox("Hazard Type", hazard_opts)
        sel_level = st.sidebar.selectbox("Intensity", levels, index=2)

        if st.sidebar.button("Run Simulation", type="primary"):
            run_params = {
                "type": "hazard",
                "country": sel_country,
                "hazard": sel_hazard,
                "level": sel_level,
            }

    else:  # Manual Mode
        st.sidebar.subheader("Manual Shock Parameters")
        all_countries = get_country_list(inputs.node_labels)
        sel_country = st.sidebar.selectbox("Target Country", all_countries)

        avail_sectors = get_sector_list(inputs.node_labels, sel_country)
        sel_sectors = st.sidebar.multiselect(
            "Target Sectors",
            avail_sectors,
            format_func=lambda x: format_sector_label(x, inputs.sector_decoder),
        )

        magnitude = st.sidebar.slider("Supply Shock Magnitude (%)", 0.0, 50.0, 5.0, 0.5)

        if st.sidebar.button("Run Simulation", type="primary"):
            run_params = {
                "type": "manual",
                "country": sel_country,
                "sectors": sel_sectors if sel_sectors else None,
                "pct": magnitude,
            }

    # --- Execution Logic ---
    if run_params:
        with st.spinner("Running Simulation..."):
            try:
                # 1. Determine Shock
                if run_params["type"] == "hazard":
                    phi = shock_scalar(
                        pct_table,
                        country_iso2=run_params["country"],
                        hazard=run_params["hazard"],
                        intensity_level=run_params["level"],
                        clamp_max=0.50,
                    )
                    supply_shock_pct = phi * 100.0
                    st.session_state["scenario_name"] = (
                        f"{run_params['level'].title()} {run_params['hazard']} in {run_params['country']}"
                    )
                    st.session_state["shock_val"] = phi

                    scenario_args = dict(
                        supply_country_codes=[run_params["country"]],
                        supply_sector_codes=None,
                        supply_shock_pct=supply_shock_pct,
                        demand_shock_pct=0.0,
                    )
                else:
                    supply_shock_pct = run_params["pct"]
                    st.session_state["shock_val"] = supply_shock_pct / 100.0
                    target = (
                        "All Sectors"
                        if not run_params["sectors"]
                        else f"{len(run_params['sectors'])} Sectors"
                    )
                    st.session_state["scenario_name"] = (
                        f"Manual Shock ({supply_shock_pct}%) on {run_params['country']} - {target}"
                    )

                    scenario_args = dict(
                        supply_country_codes=[run_params["country"]],
                        supply_sector_codes=run_params["sectors"],
                        supply_shock_pct=supply_shock_pct,
                        demand_shock_pct=0.0,
                    )

                # 2. Run Model
                scenario_args.update(
                    {
                        "gamma": 0.05,
                        "max_iter": 200,
                        "tol": 1e-4,
                        "return_history": False,
                    }
                )
                results = model.run(**scenario_args)

                # 3. Post-Process (Prepare Bundle)
                pp = postprocess_results(
                    node_labels=inputs.node_labels,
                    Z0=inputs.Z,
                    X0=inputs.X,
                    Z1=results["Z_final"],
                    X1=results["X_supply_final"],
                    FD_post=results.get("FD_post_final"),
                    sector_name_map=inputs.sector_decoder,
                    linkage_metric="A",
                    top_k_links=20,
                )

                bundle = build_dashboard_bundle(pp)

                # Apply OE Branding with specific color logic
                for fig_key in bundle.figures:
                    theme = None
                    if fig_key in ["top_sectors", "links_weakened", "top_countries"]:
                        theme = "primary"
                    elif fig_key == "links_strengthened":
                        theme = "secondary"

                    bundle.figures[fig_key] = apply_oe_branding(
                        bundle.figures[fig_key], theme_color=theme
                    )

                st.session_state["bundle"] = bundle
                st.session_state["has_run"] = True

            except Exception as e:
                st.error(f"Simulation Failed: {e}")

    # --- Dashboard Display ---
    if st.session_state.get("has_run"):
        bundle = st.session_state["bundle"]

        st.divider()
        st.subheader(f"Results: {st.session_state['scenario_name']}")

        # 1. KPIs
        X_loss = bundle.meta["X_loss_abs_total"]
        X_loss_pct = bundle.meta["X_loss_pct_total"]
        VA_loss = bundle.meta["VA_loss_abs_total"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Effective Shock (φ)", f"{st.session_state['shock_val']:.2%}")
        col2.metric("Total Output Loss", f"€{X_loss / 1e3:,.1f} B")
        col3.metric("Total VA Loss", f"€{VA_loss / 1e3:,.1f} B")
        # Relative Impact is already scaled (e.g. 5.0 for 5%)
        col4.metric("Relative Impact", f"{X_loss_pct:.2f}%")

        # 2. Tabs for Viz
        tab_map, tab_sectors, tab_links, tab_data = st.tabs(
            [
                "🗺️ Geographic Impact",
                "🏭 Sectoral Impact",
                "🔗 Supply Chain",
                "📋 Data Explorer",
            ]
        )

        with tab_map:
            col_map, col_bar = st.columns([3, 2])

            with col_map:
                # 1. Geographic Impact Map
                map_fig = bundle.figures["country_map"]
                # Default to pan and enable click selection for better UX
                map_fig.update_layout(dragmode="pan", clickmode="event+select")

                selection = st.plotly_chart(
                    map_fig,
                    use_container_width=True,
                    on_select="rerun",
                    key="map_selector",
                )

            with col_bar:
                # 2. Dynamic Top Countries Bar Chart
                selected_iso3 = []
                if selection and "selection" in selection:
                    points = selection["selection"].get("points", [])
                    selected_iso3 = [
                        p.get("location") for p in points if p.get("location")
                    ]

                df_country = bundle.tables["country"].copy()
                df_country["iso3"] = df_country["country"].map(iso2_to_iso3)

                if selected_iso3:
                    df_filtered = df_country[df_country["iso3"].isin(selected_iso3)]
                    st.caption(f"🎯 Showing {len(df_filtered)} countries active in map")
                else:
                    df_filtered = df_country
                    st.caption("🔍 Showing top impacted countries (global)")

                # Re-generate bar chart figure dynamically based on filter
                fig_top = plot_top_countries(
                    df_filtered,
                    metric="loss_abs",
                    top_k=20,
                    title="Country Impact Breakdown",
                )

                # Re-apply branding theme
                fig_top = apply_oe_branding(fig_top, theme_color="primary")

                st.plotly_chart(fig_top, use_container_width=True)

        with tab_sectors:
            st.plotly_chart(bundle.figures["top_sectors"], use_container_width=True)

        with tab_links:
            heatmap_mode = st.segmented_control(
                "Heatmap perspective",
                options=["absolute", "percentage"],
                default="absolute",
                key="supply_chain_heatmap_mode",
            )
            heatmap_key = f"supply_chain_heatmap_{heatmap_mode}"
            st.plotly_chart(bundle.figures[heatmap_key], use_container_width=True)

            col_l, col_r = st.columns(2)
            if "links_weakened" in bundle.figures:
                with col_l:
                    st.plotly_chart(
                        bundle.figures["links_weakened"], use_container_width=True
                    )
            if "links_strengthened" in bundle.figures:
                with col_r:
                    st.plotly_chart(
                        bundle.figures["links_strengthened"], use_container_width=True
                    )

        with tab_data:
            st.markdown("### Detailed Node Impacts")

            # Format dataframe for better readability
            nodes_df = bundle.tables["nodes"]

            column_config = {
                "loss_pct": st.column_config.NumberColumn(
                    "Output Loss (%)", format="%.2f%%", help="Relative loss in output"
                ),
                "VA_loss_pct": st.column_config.NumberColumn(
                    "VA Loss (%)", format="%.2f%%", help="Relative loss in Value Added"
                ),
                "loss_abs": st.column_config.NumberColumn(
                    "Output Loss (M€)",
                    format="€ %.1f",
                ),
                "VA_loss_abs": st.column_config.NumberColumn(
                    "VA Loss (M€)",
                    format="€ %.1f",
                ),
                "geo_name": "Province",
                "sector_name": "Sector",
            }

            st.dataframe(
                nodes_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
            )

            csv = bundle.tables["nodes"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results (CSV)", csv, "simulation_results.csv", "text/csv"
            )

    else:
        st.info("👈 Configure and Run a simulation to view results.")


if __name__ == "__main__":
    main()
