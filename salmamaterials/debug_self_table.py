import pandas as pd
import numpy as np
import os

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def is_m(a, b, t):
    return np.abs(a - b) <= t

def run_debug():
    threshold = 0.25
    
    # Load baselines
    p_v = pd.read_csv('perceived_valence_summary.csv')
    p_a = pd.read_csv('perceived_arousal_summary.csv')
    i_v = pd.read_csv('induced_valence_summary.csv')
    i_a = pd.read_csv('induced_arousal_summary.csv')
    e_v_vid = pd.read_csv('final_expressed_valence_VIDEO.csv')
    e_a_vid = pd.read_csv('final_expressed_arousal_VIDEO.csv')
    e_v_aud = pd.read_csv('final_expressed_valence_AUDIO.csv')
    e_a_aud = pd.read_csv('final_expressed_arousal_AUDIO.csv')

    # Standardize
    for df in [p_v, p_a, i_v, i_a, e_v_vid, e_a_vid, e_v_aud, e_a_aud]:
        df['clip_id'] = clean_id(df['clip_id'])

    # --- VIDEO VALENCE ---
    m_v_vid = pd.merge(p_v[['clip_id', 'avg_valence']], e_v_vid[['clip_id', 'expressed_valence']], on='clip_id')
    m_v_vid = pd.merge(m_v_vid, i_v[['clip_id', 'avg_valence']], on='clip_id', suffixes=('', '_induced'))
    m_v_vid.columns = ['id', 'p', 'e', 'i']
    
    h1 = ((is_m(m_v_vid.p, m_v_vid.e, threshold)) & (is_m(m_v_vid.p, m_v_vid.i, threshold))).sum()
    h2 = ((is_m(m_v_vid.p, m_v_vid.i, threshold)) & (~is_m(m_v_vid.p, m_v_vid.e, threshold))).sum()
    h3 = ((is_m(m_v_vid.p, m_v_vid.e, threshold)) & (~is_m(m_v_vid.p, m_v_vid.i, threshold))).sum()
    h4 = len(m_v_vid) - (h1+h2+h3)
    
    print(f"SELF | VIDEO | VALENCE (tau=0.25): {h1}, {h2}, {h3}, {h4} | Total: {len(m_v_vid)}")

    # --- AUDIO VALENCE ---
    m_v_aud = pd.merge(p_v[['clip_id', 'avg_valence']], e_v_aud[['clip_id', 'expressed_valence']], on='clip_id')
    m_v_aud = pd.merge(m_v_aud, i_v[['clip_id', 'avg_valence']], on='clip_id', suffixes=('', '_induced'))
    m_v_aud.columns = ['id', 'p', 'e', 'i']
    
    h1_a = ((is_m(m_v_aud.p, m_v_aud.e, threshold)) & (is_m(m_v_aud.p, m_v_aud.i, threshold))).sum()
    h2_a = ((is_m(m_v_aud.p, m_v_aud.i, threshold)) & (~is_m(m_v_aud.p, m_v_aud.e, threshold))).sum()
    h3_a = ((is_m(m_v_aud.p, m_v_aud.e, threshold)) & (~is_m(m_v_aud.p, m_v_aud.i, threshold))).sum()
    h4_a = len(m_v_aud) - (h1_a+h2_a+h3_a)
    
    print(f"SELF | AUDIO | VALENCE (tau=0.25): {h1_a}, {h2_a}, {h3_a}, {h4_a} | Total: {len(m_v_aud)}")

if __name__ == "__main__":
    run_debug()
