#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 21:08:33 2025

@author: srinivas
"""

import streamlit as st

# Set page title and icon
st.set_page_config(page_title="TJR Execution & Mindset Tool", page_icon="🧠")

st.title("🛡️ TJR Strategy & Mindset Checklist")
st.markdown("---")

# Section 1: 1H & 1m Context (The Technicals)
st.header("Step 1: Technical Confluences")
col1, col2 = st.columns(2)

with col1:
    sweep_1h = st.checkbox("1HR Liquidity Sweep (Major High/Low)")
    crit_choch = st.checkbox("1m Change of Character (Choch)")

with col2:
    crit_sweep_1m = st.checkbox("1m Internal Liquidity Sweep (Inducement)")
    crit_fvg = st.checkbox("1m FVG or IFVG Entry Zone")

# Section 2: The "Anti-Gambling" Filter
st.header("Step 2: Execution Quality")
displacement = st.checkbox("Is there strong DISPLACEMENT? (No weak candles)")
news_checked = st.checkbox("No high-impact news in the next 30 mins")

# Section 3: Psychological Circuit Breaker (The "Account Saver")
st.header("Step 3: Mental Hard-Stop")
st.info("💡 Remember: One impulsive click can reset months of work. Protect the Evaluation.")

# Specific psychological triggers
mindset_1 = st.checkbox("I am NOT revenge trading or 'trying to make back' a loss")
mindset_2 = st.checkbox("I accept that if this trade hits SL, I will NOT open another one immediately")
mindset_3 = st.checkbox("I have remembered my blown accounts and the pain of starting over")
mindset_4 = st.checkbox("I am focused on the PAYOUT, not the 'excitement' of the trade")

# Logic Gate
technicals_met = sweep_1h and crit_choch and crit_sweep_1m and crit_fvg and displacement
mindset_met = mindset_1 and mindset_2 and mindset_3 and mindset_4 and news_checked

st.markdown("---")

if technicals_met and mindset_met:
    st.success("✅ SYSTEM & MINDSET ALIGNED. You have permission to trade.")
    st.balloons()
    st.write("Target: Payout Path. Risk: 0.5% - 1% max.")
elif technicals_met and not mindset_met:
    st.error("🛑 STOP. Your strategy is there, but your BRAIN is not.")
    st.warning("Even with confluences, if you are emotional, you will mismanage the trade. Step away for 15 minutes.")
else:
    st.warning("⚠️ NO TRADE. The market has not given you the TJR setup yet. Be patient.")
    if not technicals_met:
        st.write("Missing: Wait for the 1H Sweep and 1m internal structure to align.")