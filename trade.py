import streamlit as st

# Set page title and icon
st.set_page_config(page_title="TJR Execution Tool", page_icon="📈")

st.title("🛡️ TJR Strategy Execution Checklist")
st.markdown("---")

# Section 1: HTF Context (The Setup)
st.header("Step 1: Higher Timeframe Setup")
col_htf1, col_htf2 = st.columns(2)

with col_htf1:
    htf_bias = st.radio("Daily/4H Bias Direction", ["Bullish", "Bearish", "Unclear"])
    news_checked = st.checkbox("Economic Calendar Clear (No Red Folders)")

with col_htf2:
    # STEP 1: 1HR LIQUIDITY SWEEP
    sweep_1h = st.checkbox("1HR Liquidity Sweep Confirmed (HTF POI)")
    st.caption("Price swept a 1H High/Low or Session High/Low")

# Section 2: LTF Refinement (The Execution)
st.header("Step 2: 1-Minute Precision Entry")
col1, col2 = st.columns(2)

with col1:
    # STEP 2: 1 MIN CHOCH
    crit_choch = st.checkbox("1m Change of Character (Choch)")
    # STEP 3: 1 MIN LIQUIDITY SWEEP
    crit_sweep_1m = st.checkbox("1m Internal Liquidity Sweep")

with col2:
    # STEP 4: FVG or IFVG
    crit_fvg = st.checkbox("1m FVG or IFVG present for entry")
    displacement = st.checkbox("Displacement: Strong candle body move?")

# Section 3: Risk Management
st.header("Step 3: Risk Management")
risk_pct = st.number_input("Risk % for this trade", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
rr_ratio = st.number_input("Target Risk:Reward Ratio (Minimum 1:2)", value=2.0)

# The Logic Gate - All your steps must be True
all_criteria_met = (
    sweep_1h and 
    crit_choch and 
    crit_sweep_1m and 
    crit_fvg and 
    news_checked and 
    displacement and
    (htf_bias != "Unclear")
)

st.markdown("---")

if all_criteria_met:
    st.success("✅ ALL TJR CRITERIA MET. Execute the trade.")
    st.balloons()
    st.info(f"Execution: Risking {risk_pct}% | Target RR: {rr_ratio}:1")
else:
    st.warning("⚠️ STOP. Criteria not met. Do not enter.")
    st.write("Missing requirements:")
    if not sweep_1h: st.write("- ❌ Missing 1HR Liquidity Sweep.")
    if not crit_choch: st.write("- ❌ Missing 1m Change of Character (Choch).")
    if not crit_sweep_1m: st.write("- ❌ Missing 1m Internal Liquidity Sweep.")
    if not crit_fvg: st.write("- ❌ Missing 1m FVG / IFVG for entry.")
    if not news_checked: st.write("- ❌ Check for high-impact news before trading.")