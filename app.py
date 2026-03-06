import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import time
import re

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Analyzer · Parth Tyagi",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ══ DARK BLACK + PURPLE THEME ══ */
:root {
  --bg:       #09090f;
  --bg2:      #0f0f1a;
  --bg3:      #13131f;
  --surface:  #16162a;
  --surface2: #1c1c30;
  --purple:   #9b5de5;
  --purple2:  #c084fc;
  --purple3:  #7c3aed;
  --purple4:  #4c1d95;
  --glow:     rgba(155,93,229,.35);
  --glow2:    rgba(192,132,252,.2);
  --border:   rgba(155,93,229,.25);
  --border2:  rgba(155,93,229,.12);
  --text:     #e8e8f0;
  --muted:    #6b6b8a;
  --muted2:   #9090b0;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 20% 10%,  rgba(124,58,237,.18) 0%, transparent 45%),
        radial-gradient(ellipse at 80% 90%,  rgba(155,93,229,.12) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 50%,  rgba(76,29,149,.08)  0%, transparent 70%),
        var(--bg) !important;
    min-height: 100vh;
}

[data-testid="block-container"] {
    padding: 0 !important;
    max-width: 1240px !important;
    margin: 0 auto !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--purple3); border-radius: 99px; }

/* ── KEYFRAMES ── */
@keyframes heroIn {
  from { opacity:0; transform:translateY(-28px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(18px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn { from{opacity:0} to{opacity:1} }
@keyframes shimmer {
  0%,100% { background-position:0% center; }
  50%      { background-position:200% center; }
}
@keyframes cardIn {
  from { opacity:0; transform:translateY(24px) rotate3d(.5,1,0,8deg); }
  to   { opacity:1; transform:none; }
}
@keyframes tagPop {
  from { transform:scale(.55); opacity:0; }
  to   { transform:scale(1);   opacity:1; }
}
@keyframes barFill { from { width:0%; } }
@keyframes glowPulse {
  0%,100% { box-shadow:0 4px 20px rgba(124,58,237,.4),0 0 0 1px rgba(155,93,229,.3); }
  50%      { box-shadow:0 4px 32px rgba(124,58,237,.7),0 0 40px rgba(155,93,229,.25); }
}
@keyframes orbFloat {
  0%,100% { transform:translateY(0) scale(1); opacity:.7; }
  50%      { transform:translateY(-22px) scale(1.06); opacity:1; }
}
@keyframes scanline {
  0%   { top:-4px; opacity:0; }
  10%  { opacity:1; }
  90%  { opacity:1; }
  100% { top:100%; opacity:0; }
}
@keyframes pip {
  0%,100% { transform:scale(1); box-shadow:0 0 6px var(--purple); }
  50%      { transform:scale(1.9); box-shadow:0 0 22px var(--purple); }
}
@keyframes borderGlow {
  0%,100% { border-color:rgba(155,93,229,.25); }
  50%      { border-color:rgba(155,93,229,.65); }
}

/* ── HERO ── */
.hero-wrap {
  background: linear-gradient(135deg, #0a0a1e 0%, #110d28 50%, #0a0a1e 100%);
  border-bottom: 1px solid var(--border);
  padding: 56px 64px 50px;
  position: relative; overflow: hidden;
  animation: heroIn .8s cubic-bezier(.22,1,.36,1) both;
}
/* grid lines */
.hero-wrap::before {
  content: '';
  position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(155,93,229,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(155,93,229,.055) 1px, transparent 1px);
  background-size: 44px 44px;
  animation: fadeIn 1.5s ease both;
}
/* scanline */
.hero-wrap::after {
  content: '';
  position: absolute; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, transparent, rgba(155,93,229,.2), rgba(192,132,252,.15), transparent);
  animation: scanline 7s linear infinite;
  pointer-events: none; z-index: 2;
}
.hero-orb1 {
  position: absolute; width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(circle, rgba(124,58,237,.22) 0%, transparent 70%);
  top: -100px; right: 60px; pointer-events: none;
  animation: orbFloat 7s ease infinite;
}
.hero-orb2 {
  position: absolute; width: 220px; height: 220px; border-radius: 50%;
  background: radial-gradient(circle, rgba(192,132,252,.14) 0%, transparent 70%);
  bottom: -70px; left: 180px; pointer-events: none;
  animation: orbFloat 9s ease infinite reverse;
}
.hero-inner { position: relative; z-index: 1; }

.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 16px 5px 8px;
  background: rgba(155,93,229,.1);
  border: 1px solid rgba(155,93,229,.35);
  border-radius: 99px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; letter-spacing: .14em; color: var(--purple2);
  margin-bottom: 22px;
  animation: fadeUp .6s ease .1s both;
}
.hero-badge .bdot {
  width: 18px; height: 18px; border-radius: 50%;
  background: linear-gradient(135deg, var(--purple3), var(--purple));
  color: #fff; display: inline-flex; align-items: center;
  justify-content: center; font-size: .48rem; font-weight: 900;
  box-shadow: 0 0 10px var(--glow);
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .63rem; letter-spacing: .32em; text-transform: uppercase;
  color: var(--muted2); margin-bottom: 14px;
  display: flex; align-items: center; gap: 12px;
  animation: fadeUp .6s ease .15s both;
}
.hero-eyebrow::before {
  content: ''; width: 28px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--purple));
}
.hero-pip {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--purple); margin-right: 8px;
  animation: pip 2.4s ease infinite;
}
.hero-title {
  font-size: clamp(2.6rem, 5.5vw, 5rem);
  font-weight: 900; letter-spacing: -.04em; line-height: .9;
  color: var(--text); margin-bottom: 20px;
  animation: fadeUp .7s ease .2s both;
}
.hero-title .grad {
  background: linear-gradient(110deg, #fff 0%, var(--purple2) 40%, var(--purple) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: shimmer 5s ease infinite;
}
.hero-sub {
  font-size: 1rem; color: var(--muted2); line-height: 1.78;
  max-width: 560px; animation: fadeUp .7s ease .3s both;
}
.hero-sub strong { color: var(--text); font-weight: 700; }

/* ── STEP LABEL ── */
.step-label {
  display: inline-flex; align-items: center; gap: 8px;
  background: linear-gradient(135deg, rgba(124,58,237,.18), rgba(155,93,229,.08));
  color: var(--purple2);
  border: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace;
  font-size: .61rem; letter-spacing: .16em; text-transform: uppercase;
  padding: 6px 16px; margin-bottom: 14px; border-radius: 4px;
  animation: fadeUp .5s ease both;
}

/* ── SECTION CARD ── */
.section-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--purple);
  padding: 28px 32px; margin-bottom: 20px; border-radius: 8px;
  box-shadow: 0 4px 28px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.03);
  animation: fadeUp .6s ease both;
}

/* ── FIELD LABEL ── */
.field-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; letter-spacing: .24em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 10px; display: block;
}

/* ── TEXTAREA: visible cursor + glow ── */
.stTextArea textarea {
  font-family: 'Outfit', sans-serif !important;
  font-size: .95rem !important;
  border: 1.5px solid rgba(155,93,229,.28) !important;
  border-radius: 8px !important;
  background: #0d0d1e !important;
  color: #e8e8f0 !important;
  caret-color: #c084fc !important;
  caret-width: 2px !important;
  padding: 14px 16px !important;
  box-shadow: none !important;
  transition: border-color .25s, box-shadow .25s, background .25s !important;
  resize: vertical !important;
  line-height: 1.7 !important;
}
.stTextArea textarea::placeholder {
  color: rgba(155,93,229,.3) !important;
  font-style: italic !important;
}
.stTextArea textarea:hover {
  border-color: rgba(155,93,229,.45) !important;
}
.stTextArea textarea:focus {
  border-color: var(--purple) !important;
  box-shadow: 0 0 0 3px rgba(155,93,229,.18), 0 0 20px rgba(155,93,229,.1) !important;
  outline: none !important;
  background: #0f0f22 !important;
  caret-color: #c084fc !important;
}

/* ── SELECTBOX: show selected value clearly ── */
.stSelectbox > div > div {
  border: 1.5px solid rgba(155,93,229,.28) !important;
  border-radius: 8px !important;
  background: #0d0d1e !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: .9rem !important;
  color: #e8e8f0 !important;
  transition: border-color .2s, box-shadow .2s !important;
}
/* the selected value text */
.stSelectbox [data-testid="stSelectbox"] div,
.stSelectbox > div > div > div,
.stSelectbox > div > div > div > div {
  color: #c084fc !important;
  font-weight: 600 !important;
  font-family: 'Outfit', sans-serif !important;
}
.stSelectbox > div > div:focus-within {
  border-color: var(--purple) !important;
  box-shadow: 0 0 0 3px rgba(155,93,229,.18) !important;
}
.stSelectbox > div > div svg path,
.stSelectbox > div > div svg polyline {
  stroke: var(--purple2) !important;
}
/* dropdown list */
[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
  background: #14142a !important;
  border: 1px solid rgba(155,93,229,.3) !important;
  border-radius: 8px !important;
  box-shadow: 0 20px 60px rgba(0,0,0,.7) !important;
}
[data-baseweb="menu"] li {
  background: transparent !important;
  color: #e8e8f0 !important;
  font-family: 'Outfit', sans-serif !important;
  transition: background .15s !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] {
  background: rgba(155,93,229,.18) !important;
  color: var(--purple2) !important;
}
.stTextArea label, .stSelectbox label { display: none !important; }

/* ── BUTTON ── */
.stButton > button {
  background: linear-gradient(135deg, var(--purple3) 0%, var(--purple) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 800 !important;
  font-size: .82rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  padding: 14px 36px !important;
  width: 100% !important;
  transition: all .3s !important;
  animation: glowPulse 3s ease infinite !important;
  position: relative !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 10px 40px rgba(124,58,237,.65), 0 0 60px rgba(155,93,229,.25) !important;
}
.stButton > button:active { transform: translateY(1px) !important; }

/* ── 3D CARD (purple checkerboard) ── */
.card3-wrap { perspective: 1000px; margin-bottom: 14px; }
.card3 {
  padding-top: 46px;
  border: 1.5px solid rgba(155,93,229,.35);
  transform-style: preserve-3d;
  background:
    linear-gradient(135deg,#0000 18.75%,rgba(124,58,237,.13) 0 31.25%,#0000 0),
    repeating-linear-gradient(45deg,rgba(124,58,237,.13) -6.25% 6.25%,#0e0e20 0 18.75%);
  background-size: 44px 44px;
  background-position: 0 0, 0 0;
  background-color: #0c0c1c;
  position: relative;
  box-shadow: 0 18px 40px rgba(0,0,0,.55), 0 0 18px rgba(155,93,229,.08);
  transition: all .5s ease-in-out;
  animation: cardIn .6s ease both;
  border-radius: 8px; overflow: hidden;
}
.card3:hover {
  background-position: -80px 80px, -80px 80px;
  transform: rotate3d(.5,1,0,24deg);
  box-shadow: 0 40px 72px rgba(0,0,0,.65), 0 0 40px rgba(155,93,229,.28);
  border-color: var(--purple);
}
.card3-body {
  background: linear-gradient(135deg, #151525, #1a1030);
  padding: 22px 22px 20px;
  transform-style: preserve-3d;
  border-top: 1px solid rgba(155,93,229,.18);
}
.card3-icon {
  position: absolute; top: 13px; left: 16px; font-size: 1.3rem;
  transform: translate3d(0,0,60px); transition: transform .5s ease;
  filter: drop-shadow(0 0 8px rgba(155,93,229,.4));
}
.card3:hover .card3-icon { transform: translate3d(0,0,84px); }
.card3-datebox {
  position: absolute; top: 10px; right: 12px;
  height: 44px; width: 44px;
  background: linear-gradient(135deg, var(--surface2), #1a1030);
  border: 1px solid rgba(155,93,229,.4);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 1px;
  transform: translate3d(0,0,72px);
  box-shadow: 0 8px 20px rgba(0,0,0,.5), 0 0 12px rgba(155,93,229,.18);
  transition: transform .5s ease; border-radius: 4px;
}
.card3:hover .card3-datebox { transform: translate3d(0,0,96px); }
.card3-datebox .db-t {
  font-size: .44rem; font-weight: 700; text-transform: uppercase;
  color: var(--muted); letter-spacing: .08em;
}
.card3-datebox .db-v {
  font-size: .82rem; font-weight: 900; color: var(--purple2);
  font-family: 'JetBrains Mono', monospace;
}
.card3-title {
  display: block; color: #fff; font-size: .88rem; font-weight: 900;
  letter-spacing: .02em; margin-bottom: 3px;
  transform: translate3d(0,0,48px); transition: transform .5s ease;
}
.card3:hover .card3-title { transform: translate3d(0,0,66px); }
.card3-sub {
  display: block; color: var(--muted);
  font-size: .66rem; font-family: 'JetBrains Mono', monospace;
  transform: translate3d(0,0,28px); transition: transform .5s ease;
}
.card3:hover .card3-sub { transform: translate3d(0,0,46px); }

/* ── JOB RESULT CARD ── */
.jcard {
  border: 1px solid var(--border2);
  background: var(--surface);
  margin-bottom: 16px; border-radius: 10px; overflow: hidden;
  transition: all .35s ease;
  animation: fadeUp .5s ease both;
  position: relative;
  box-shadow: 0 4px 20px rgba(0,0,0,.45);
}
.jcard:hover {
  transform: translateY(-4px);
  border-color: rgba(155,93,229,.5);
  box-shadow: 0 12px 42px rgba(0,0,0,.55), 0 0 28px rgba(155,93,229,.14);
}
.jcard-head {
  background: linear-gradient(135deg, #160e2e, #120d24);
  padding: 16px 20px 16px 64px;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid rgba(155,93,229,.18);
  gap: 12px;
}
.jcard-rank {
  position: absolute; top: 0; left: 0;
  width: 48px; height: 100%;
  background: linear-gradient(180deg, var(--purple3), var(--purple4));
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem; font-weight: 900; color: #fff;
  box-shadow: 2px 0 14px rgba(124,58,237,.4);
}
.jcard-title {
  font-weight: 900; font-size: 1.05rem; color: #fff;
  letter-spacing: -.01em; line-height: 1.2;
  flex: 1;
}
.jcard-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.1rem; font-weight: 900;
  background: linear-gradient(110deg, var(--purple2), var(--purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; white-space: nowrap;
}
.bar-outer { height: 3px; background: rgba(155,93,229,.08); margin-left: 48px; }
.bar-inner {
  height: 100%;
  background: linear-gradient(90deg, var(--purple3), var(--purple2));
  animation: barFill 1.3s cubic-bezier(.22,1,.36,1) both;
  box-shadow: 0 0 8px var(--glow);
}
.jcard-body { padding: 16px 20px 16px 64px; }

/* ── SKILL TAGS ── */
.stag {
  display: inline-block; padding: 4px 12px; margin: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .67rem; font-weight: 600; border-radius: 4px;
  animation: tagPop .35s cubic-bezier(.34,1.56,.64,1) both;
}
.stag.got {
  background: rgba(124,58,237,.16);
  color: var(--purple2);
  border: 1px solid rgba(155,93,229,.3);
}
.stag.got::before { content: '✓ '; color: var(--purple); }
.stag.miss {
  background: rgba(239,68,68,.09);
  color: #fca5a5;
  border: 1px solid rgba(239,68,68,.22);
}
.stag.miss::before { content: '× '; color: #ef4444; }

/* ── METRIC STRIP ── */
.metric-strip {
  display: flex; border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden;
  margin-bottom: 20px; animation: fadeUp .5s ease both;
  box-shadow: 0 4px 22px rgba(0,0,0,.45);
}
.metric-cell {
  flex: 1; padding: 16px 14px;
  border-right: 1px solid var(--border2);
  text-align: center; background: var(--surface);
  transition: background .2s;
}
.metric-cell:last-child { border-right: none; }
.metric-cell:hover { background: var(--surface2); }
.metric-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.45rem; font-weight: 900;
  background: linear-gradient(110deg, #fff, var(--purple2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; display: block;
}
.metric-lbl {
  font-size: .6rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--muted); display: block; margin-top: 3px;
}

/* ── ROADMAP ── */
.roadmap-row {
  display: flex; align-items: center; gap: 14px;
  padding: 10px 0; border-bottom: 1px solid rgba(155,93,229,.07);
}
.roadmap-skill {
  font-family: 'JetBrains Mono', monospace;
  font-size: .73rem; font-weight: 700; color: var(--text);
  min-width: 150px;
}
.roadmap-bar {
  flex: 1; background: rgba(155,93,229,.1); height: 4px;
  border-radius: 99px; overflow: hidden;
}
.roadmap-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--purple3), var(--purple2));
  border-radius: 99px;
  animation: barFill 1.1s cubic-bezier(.22,1,.36,1) both;
  box-shadow: 0 0 6px var(--glow);
}
.roadmap-meta {
  font-size: .64rem; color: var(--muted); min-width: 90px; text-align: right;
}

/* ── EMPTY STATE ── */
.empty-state {
  text-align: center; padding: 60px 24px;
  border: 1px dashed rgba(155,93,229,.2);
  border-radius: 12px;
  background: rgba(155,93,229,.03);
  animation: fadeUp .5s ease both;
}
.empty-icon {
  font-size: 3rem; margin-bottom: 12px; opacity:.5;
  filter: drop-shadow(0 0 14px rgba(155,93,229,.4));
}
.empty-txt { font-size:.9rem; color:var(--muted2); font-weight:500; line-height:1.65; }

/* ── ALERT ── */
.alert {
  background: linear-gradient(135deg, rgba(124,58,237,.13), rgba(155,93,229,.07));
  color: var(--purple2); padding: 12px 18px; margin: 10px 0;
  font-size: .86rem; border-radius: 8px;
  border: 1px solid rgba(155,93,229,.28); border-left: 3px solid var(--purple);
  animation: fadeUp .4s ease both;
  font-family: 'JetBrains Mono', monospace;
}
.alert.warn {
  background: rgba(251,191,36,.07); color: #fcd34d;
  border-color: rgba(251,191,36,.28); border-left-color: #f59e0b;
}

/* ── SELECTION PILL ── */
.sel-pill {
  margin-top: 8px; padding: 10px 14px;
  background: rgba(155,93,229,.07);
  border: 1px solid rgba(155,93,229,.18);
  border-radius: 6px;
  display: flex; align-items: center; gap: 6px;
  animation: fadeIn .3s ease both;
}
.sel-pill-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .63rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted);
}
.sel-pill-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: .75rem; font-weight: 700; color: var(--purple2);
  background: rgba(155,93,229,.12);
  padding: 2px 10px; border-radius: 4px;
  border: 1px solid rgba(155,93,229,.25);
}

/* ── FOOTER ── */
.footer {
  background: linear-gradient(135deg, #0a0a18, #0f0a20);
  padding: 22px 64px;
  display: flex; align-items: center; justify-content: space-between;
  border-top: 1px solid var(--border2); margin-top: 56px;
}
.footer-l {
  font-family: 'JetBrains Mono', monospace;
  font-size: .63rem; color: var(--muted); letter-spacing: .1em;
}
.footer-r { font-size: .7rem; color: var(--muted); font-style: italic; }

/* stagger delays */
.d1{animation-delay:.08s}.d2{animation-delay:.18s}
.d3{animation-delay:.28s}.d4{animation-delay:.38s}
</style>
""", unsafe_allow_html=True)


# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("DataAnalyst.csv")
        df.columns = df.columns.str.strip().str.lower()

        # ── find TITLE column first (higher priority) ──
        title_col = None
        for c in df.columns:
            if any(k == c or c.startswith(k) for k in ['job_title','title','role','position','job name','jobtitle']):
                title_col = c; break
        if not title_col:
            for c in df.columns:
                if 'title' in c or 'role' in c or 'position' in c:
                    title_col = c; break

        # ── find SKILLS column (must NOT be the title column) ──
        skills_col = None
        for c in df.columns:
            if c == title_col:
                continue
            if any(k in c for k in ['skill','desc','qualif','require','key','keyword','responsibility','summary']):
                skills_col = c; break
        if not skills_col:
            # pick the longest-text column that isn't the title
            best, best_len = None, 0
            for c in df.columns:
                if c == title_col:
                    continue
                avg_len = df[c].fillna('').astype(str).str.len().mean()
                if avg_len > best_len:
                    best_len = avg_len
                    best = c
            skills_col = best or df.columns[0]

        df['skills_text'] = df[skills_col].fillna('').astype(str)
        df['job_title']   = df[title_col].fillna('Unknown Role').astype(str) if title_col else \
                            df.index.map(lambda i: f"Role #{i+1}")

        return df[['job_title','skills_text']].drop_duplicates('job_title').reset_index(drop=True)
    except Exception:
        rows = [
            ("Data Analyst",
             "python sql pandas numpy matplotlib seaborn excel statistics visualization reporting dashboards power bi tableau"),
            ("Machine Learning Engineer",
             "python machine learning scikit-learn tensorflow pytorch xgboost feature engineering model deployment docker api mlops"),
            ("Data Scientist",
             "python r statistics machine learning regression classification clustering nlp pandas numpy scipy matplotlib jupyter research"),
            ("Business Intelligence Analyst",
             "sql excel tableau power bi data visualization business analysis kpi reporting dashboards analytics"),
            ("NLP Engineer",
             "python nlp spacy nltk transformers bert gpt text classification sentiment analysis huggingface pytorch"),
            ("Computer Vision Engineer",
             "python opencv tensorflow pytorch cnn image classification object detection deep learning gpu"),
            ("MLOps Engineer",
             "docker kubernetes airflow mlflow ci cd model deployment monitoring python devops git cloud aws"),
            ("Analytics Engineer",
             "sql dbt airflow spark data pipeline etl data warehouse bigquery snowflake python analytics"),
            ("AI Research Engineer",
             "python pytorch tensorflow research deep learning optimization algorithms gradient gpu distributed training mathematics"),
            ("Backend ML Engineer",
             "python fastapi flask docker postgresql mongodb machine learning model serving rest api microservices"),
            ("Data Engineer",
             "python sql spark kafka airflow etl pipeline data lake warehouse bigquery aws glue pyspark"),
            ("Quantitative Analyst",
             "python r statistics probability mathematics linear algebra monte carlo simulation finance pandas numpy scipy"),
        ]
        df = pd.DataFrame(rows, columns=['job_title','skills_text'])
        return df


@st.cache_resource
def build_model(data):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=6000)
    mat = vec.fit_transform(data['skills_text'])
    return vec, mat


def extract_skills(text):
    text = text.lower()
    multiword = [
        'machine learning','deep learning','natural language processing',
        'computer vision','data analysis','data science','feature engineering',
        'model evaluation','neural network','random forest','gradient boosting',
        'decision tree','logistic regression','linear regression','power bi',
        'big data','data pipeline','rest api','version control','time series',
        'a/b testing','data visualization','transfer learning',
    ]
    found = []
    for mw in multiword:
        if mw in text:
            found.append(mw)
            text = text.replace(mw,' ')
    stop = {'and','the','with','for','using','from','into','this','that',
            'have','will','are','was','been','also','both','all','any',
            'can','such','very','its','more','than','our','your'}
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\.]{1,22}\b', text)
    single = [t.lower() for t in tokens if t.lower() not in stop and len(t) > 2]
    return list(dict.fromkeys(found + single))


def analyze(user_text, vectorizer, matrix, df, top_n):
    uv = vectorizer.transform([user_text])
    sims = cosine_similarity(uv, matrix).flatten()
    idxs = sims.argsort()[::-1][:top_n]
    out = []
    user_skills = set(extract_skills(user_text))
    for i in idxs:
        job_skills = set(extract_skills(df['skills_text'].iloc[i]))
        matched = sorted(user_skills & job_skills)
        missing = sorted(list(job_skills - user_skills)[:12])
        out.append({
            'title':   str(df['job_title'].iloc[i]),
            'score':   round(float(sims[i])*100, 1),
            'matched': matched,
            'missing': missing,
        })
    return out


# ─── LOAD ─────────────────────────────────────────────────────────────────────
df       = load_data()
vec, mat = build_model(df)

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-orb1"></div>
  <div class="hero-orb2"></div>
  <div class="hero-inner">
    <div class="hero-badge">
      <span class="bdot">AI</span>
      NLP-Powered Career Intelligence
    </div>
    <div class="hero-eyebrow">
      <span class="hero-pip"></span>Skill Gap Analyzer
    </div>
    <div class="hero-title">
      Know Your Gaps.<br>
      <span class="grad">Close Them.</span>
    </div>
    <div class="hero-sub">
      Enter your skills and get <strong>ranked job matches</strong>,
      exact <strong>skill gap analysis</strong>, and a
      <strong>priority upskilling roadmap</strong> — powered by
      TF-IDF vectorization &amp; cosine similarity.
    </div>
  </div>
</div>
<div style="height:32px"></div>
""", unsafe_allow_html=True)

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.18], gap="large")

# ══════════════════════════════
#  LEFT PANEL
# ══════════════════════════════
with left:
    st.markdown('<div class="step-label">⬤ &nbsp;Step 01 — Your Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card"><span class="field-label">Your Skills &amp; Technologies</span></div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "skills",
        placeholder=(
            "Type or paste your skills here…\n\n"
            "e.g.  Python, SQL, Pandas, scikit-learn,\n"
            "      XGBoost, NLP, data visualization,\n"
            "      machine learning, matplotlib, git\n\n"
            "The more detail → the better your match."
        ),
        height=210,
        key="user_skills"
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<span class="field-label">Match Options</span>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        top_n = st.selectbox(
            "Top N Roles",
            options=[3, 5, 7, 10],
            index=1,
            format_func=lambda x: f"Top {x} roles",
            key="top_n_sel"
        )
    with c2:
        mode = st.selectbox(
            "Match Mode",
            options=["Standard", "Strict", "Broad"],
            index=0,
            format_func=lambda x: f"Mode: {x}",
            key="mode_sel"
        )

    # ── SELECTED VALUES DISPLAY ──
    st.markdown(f"""
    <div class="sel-pill">
      <span class="sel-pill-label">Selected →</span>
      <span class="sel-pill-val">Top {top_n} roles</span>
      <span class="sel-pill-val">Mode: {mode}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    go = st.button("🎯  Analyze My Skills →", key="go")

    # ── HOW IT WORKS ──
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="step-label d2">⬤ &nbsp;How It Works</div>', unsafe_allow_html=True)

    for i, (icon, step, title, sub) in enumerate([
        ("🔢", "01", "TF-IDF Vectorize",  "Convert skills → weighted vectors"),
        ("📐", "02", "Cosine Similarity", "Score each job role against you"),
        ("🎯", "03", "Gap Detection",     "Surface missing skills by priority"),
    ]):
        st.markdown(f"""
        <div class="card3-wrap d{i+1}">
          <div class="card3">
            <div class="card3-icon">{icon}</div>
            <div class="card3-datebox">
              <span class="db-t">Step</span>
              <span class="db-v">{step}</span>
            </div>
            <div class="card3-body">
              <span class="card3-title">{title}</span>
              <span class="card3-sub">{sub}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card3-wrap d4">
      <div class="card3">
        <div class="card3-icon">📊</div>
        <div class="card3-datebox">
          <span class="db-t">Total</span>
          <span class="db-v">{len(df)}</span>
        </div>
        <div class="card3-body">
          <span class="card3-title">Job Roles in Dataset</span>
          <span class="card3-sub">DataAnalyst.csv · live matching</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════
#  RIGHT PANEL
# ══════════════════════════════
with right:
    st.markdown('<div class="step-label d1">⬤ &nbsp;Step 02 — Results</div>', unsafe_allow_html=True)

    if not go or not user_input.strip():
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🎯</div>
          <div class="empty-txt">
            Enter your skills on the left<br>
            and press <strong style="color:#c084fc;">Analyze</strong> to see your<br>
            matched roles and skill gaps.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, (icon, val, label, sub) in zip([c1,c2,c3],[
            ("📊", str(len(df)), "Job Roles",  "in dataset"),
            ("🧠", "TF-IDF",    "Algorithm",  "vectorization"),
            ("⚡", "NLP",       "Powered",    "skill matching"),
        ]):
            with col:
                st.markdown(f"""
                <div class="card3-wrap">
                  <div class="card3">
                    <div class="card3-icon">{icon}</div>
                    <div class="card3-datebox">
                      <span class="db-t">Info</span>
                      <span class="db-v" style="font-size:.65rem;">{val[:5]}</span>
                    </div>
                    <div class="card3-body">
                      <span class="card3-title">{label}</span>
                      <span class="card3-sub">{sub}</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    else:
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="alert">⟳ &nbsp;Vectorizing skills &amp; computing similarity scores…</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.55)
        results = analyze(user_input, vec, mat, df, top_n)
        placeholder.empty()

        if not results or results[0]['score'] < 1:
            st.markdown(
                '<div class="alert warn">⚠ No strong matches. Try adding more specific technical skills.</div>',
                unsafe_allow_html=True
            )
        else:
            top = results[0]
            avg = np.mean([r['score'] for r in results])
            n_gap = len(set(s for r in results for s in r['missing']))

            # ── METRIC STRIP ──
            st.markdown(f"""
            <div class="metric-strip">
              <div class="metric-cell">
                <span class="metric-num">{top['score']}%</span>
                <span class="metric-lbl">Best Match</span>
              </div>
              <div class="metric-cell">
                <span class="metric-num">{avg:.0f}%</span>
                <span class="metric-lbl">Avg Score</span>
              </div>
              <div class="metric-cell">
                <span class="metric-num">{len(results)}</span>
                <span class="metric-lbl">Roles Found</span>
              </div>
              <div class="metric-cell">
                <span class="metric-num">{n_gap}</span>
                <span class="metric-lbl">Unique Gaps</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── JOB CARDS ──
            st.markdown('<span class="field-label">Matched Roles — Ranked by Fit</span>', unsafe_allow_html=True)

            for rank, r in enumerate(results, 1):
                bar = min(int(r['score']), 100)
                card_delay = f"animation-delay:{(rank-1)*0.1:.2f}s"

                matched_tags = ''.join(
                    f'<span class="stag got" style="animation-delay:{j*0.05:.2f}s">{s}</span>'
                    for j, s in enumerate(r['matched'][:8])
                ) or '<span style="color:#6b6b8a;font-size:.78rem;">—</span>'

                missing_tags = ''.join(
                    f'<span class="stag miss" style="animation-delay:{j*0.05:.2f}s">{s}</span>'
                    for j, s in enumerate(r['missing'][:8])
                ) or '<span style="color:#a78bfa;font-size:.78rem;">✓ No critical gaps!</span>'

                st.markdown(f"""
                <div class="jcard" style="{card_delay}">
                  <div class="jcard-rank">#{rank}</div>
                  <div class="jcard-head">
                    <span class="jcard-title">💼 {r['title']}</span>
                    <span class="jcard-score">{r['score']}%</span>
                  </div>
                  <div class="bar-outer">
                    <div class="bar-inner" style="width:{bar}%"></div>
                  </div>
                  <div class="jcard-body">
                    <div style="margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid rgba(155,93,229,.1);">
                      <span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                                   letter-spacing:.2em;text-transform:uppercase;color:#6b6b8a;">
                        ROLE MATCH
                      </span>
                      <div style="font-size:1rem;font-weight:800;color:#e8e8f0;margin-top:3px;
                                  letter-spacing:-.01em;">
                        {r['title']}
                      </div>
                    </div>
                    <div style="margin-bottom:12px;">
                      <span class="field-label" style="margin-bottom:6px;">✓ Skills You Have</span>
                      {matched_tags}
                    </div>
                    <div>
                      <span class="field-label" style="margin-bottom:6px;">× Skills to Acquire</span>
                      {missing_tags}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── ROADMAP ──
            all_miss = [s for r in results for s in r['missing']]
            top_gaps = Counter(all_miss).most_common(10)

            if top_gaps:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div class="section-card">
                  <span class="field-label">🎓 Priority Upskilling Roadmap</span>
                  <div style="color:#9090b0;font-size:.84rem;margin-bottom:16px;line-height:1.65;">
                    Skills appearing most frequently across your matched roles.<br>
                    <strong style="color:#c084fc;">Learn these first — highest ROI.</strong>
                  </div>
                """, unsafe_allow_html=True)

                for skill, cnt in top_gaps:
                    pct = int((cnt / len(results)) * 100)
                    pri = "🔴 High"   if cnt >= len(results)*.6 else \
                          "🟡 Medium" if cnt >= len(results)*.3 else "🟢 Low"
                    d_ms = top_gaps.index((skill, cnt)) * 80
                    st.markdown(f"""
                    <div class="roadmap-row">
                      <span class="roadmap-skill">{skill}</span>
                      <div class="roadmap-bar">
                        <div class="roadmap-fill"
                             style="width:{pct}%;animation-delay:{d_ms}ms"></div>
                      </div>
                      <span class="roadmap-meta">{pri} · {cnt}/{len(results)}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-l">AI JOB SKILL GAP ANALYZER · Parth Tyagi · 2026</div>
  <div class="footer-r">"Know your gaps. Close them deliberately."</div>
</div>
""", unsafe_allow_html=True)
