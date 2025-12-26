import streamlit as st

# Set page title and icon
st.set_page_config(page_title="Execution Tool", page_icon="📈")

st.title("xecution Checklist")
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

# Section 3: Psychological Circuit Breaker (The "Account Saver")
st.header("Step 3: Mental Hard-Stop")
st.info("💡 Remember: One impulsive click can reset months of work. Protect the Evaluation.")

# Specific psychological triggers
mindset_1 = st.checkbox("I am NOT revenge trading or 'trying to make back' a loss")
mindset_2 = st.checkbox("I accept that if this trade hits SL, I will NOT open another one immediately")
mindset_3 = st.checkbox("I have remembered my blown accounts and the pain of starting over")
mindset_4 = st.checkbox("I am focused on the PAYOUT, not the 'excitement' of the trade")

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
else:
    st.warning("⚠️ STOP. Criteria not met. Do not enter.")
    st.write("Missing requirements:")
    if not sweep_1h: st.write("- ❌ Missing 1HR Liquidity Sweep.")
    if not crit_choch: st.write("- ❌ Missing 1m Change of Character (Choch).")
    if not crit_sweep_1m: st.write("- ❌ Missing 1m Internal Liquidity Sweep.")
    if not crit_fvg: st.write("- ❌ Missing 1m FVG / IFVG for entry.")
    if not news_checked: st.write("- ❌ Check for high-impact news before trading.")