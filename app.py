import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import json
import requests
import gc 
from datetime import datetime
import pytz

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Flight Operations Command Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div.stButton > button:first-child { width: 100%; }
    div[data-testid="stRadio"] > label { display: none; } /* Hide radio label */
    div[data-testid="stRadio"] > div { flex-direction: row; justify-content: center; gap: 20px; } /* Horizontal Nav */
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA CONSTANTS & LOADERS
# ==========================================
AIRPORT_COORDS = {
    'ATL': (33.64, -84.42), 'BOS': (42.36, -71.00), 'BWI': (39.17, -76.66),
    'CLT': (35.21, -80.94), 'DCA': (38.85, -77.04), 'DEN': (39.85, -104.67),
    'DFW': (32.89, -97.04), 'DTW': (42.21, -83.35), 'EWR': (40.68, -74.17),
    'FLL': (26.07, -80.15), 'HNL': (21.31, -157.92), 'IAD': (38.95, -77.45),
    'IAH': (29.99, -95.33), 'JFK': (40.64, -73.77), 'LAS': (36.08, -115.15),
    'LAX': (33.94, -118.40), 'LGA': (40.77, -73.87), 'MCO': (28.43, -81.30),
    'MDW': (41.78, -87.75), 'MEM': (35.04, -89.97), 'MIA': (25.79, -80.28),
    'MSP': (44.88, -93.22), 'ORD': (41.97, -87.90), 'PHL': (39.87, -75.24),
    'PHX': (33.43, -112.01), 'PDX': (45.58, -122.59), 'SAN': (32.73, -117.19),
    'SEA': (47.45, -122.31), 'SFO': (37.61, -122.37), 'SLC': (40.78, -111.97),
    'TPA': (27.97, -82.53)
}

AIRPORT_TIMEZONES = {
    'ATL': 'America/New_York', 'BOS': 'America/New_York', 'BWI': 'America/New_York',
    'CLT': 'America/New_York', 'DCA': 'America/New_York', 'DTW': 'America/New_York',
    'EWR': 'America/New_York', 'FLL': 'America/New_York', 'IAD': 'America/New_York',
    'JFK': 'America/New_York', 'LGA': 'America/New_York', 'MCO': 'America/New_York',
    'MIA': 'America/New_York', 'PHL': 'America/New_York', 'TPA': 'America/New_York',
    'ORD': 'America/Chicago', 'MDW': 'America/Chicago', 'MSP': 'America/Chicago',
    'MEM': 'America/Chicago', 'DFW': 'America/Chicago', 'IAH': 'America/Chicago',
    'DEN': 'America/Denver', 'SLC': 'America/Denver', 'PHX': 'America/Phoenix',
    'LAX': 'America/Los_Angeles', 'SFO': 'America/Los_Angeles', 'SEA': 'America/Los_Angeles',
    'LAS': 'America/Los_Angeles', 'PDX': 'America/Los_Angeles', 'SAN': 'America/Los_Angeles',
    'HNL': 'Pacific/Honolulu'
}

@st.cache_data
def load_data():
    """OPTIMIZED LOADER V30: With RAM Management"""
    dtypes = {
        'AIRLINE': 'category', 'ORIGIN': 'category', 'DEST': 'category',
        'FL_NUMBER': 'uint16', 'DISTANCE': 'float32',
        'Route_Risk_Score': 'float32', 'Congestion': 'float32'
    }
    
    df_f = pd.read_csv('data.zip', dtype=dtypes)
    df_w = pd.read_csv('weather_data_hourly_master.csv')
    
    df_f['FL_DATE'] = pd.to_datetime(df_f['FL_DATE']).dt.date
    df_f['Join_Hour'] = (df_f['CRS_DEP_TIME'] // 100).astype('uint8')
    
    df_w['Weather_Timestamp'] = pd.to_datetime(df_w['Weather_Timestamp'])
    df_w['FL_DATE'] = df_w['Weather_Timestamp'].dt.date
    df_w['Join_Hour'] = df_w['Weather_Timestamp'].dt.hour.astype('uint8')
    
    df = pd.merge(df_f, df_w, left_on=['FL_DATE','Join_Hour','ORIGIN'], right_on=['FL_DATE','Join_Hour','ORIGIN'], how='left')
    
    if 'temp' in df.columns:
        df['temp'] = df['temp'].fillna(df['temp'].mean()).astype('float32')
        df['wspd'] = df['wspd'].fillna(df['wspd'].mean()).astype('float32')
        df['prcp'] = df['prcp'].fillna(0).astype('float32')
        df['Is_Snowing'] = ((df['temp'] < 0) & (df['prcp'] > 0)).astype('int8')

    # Load Search Index (JSON)
    try:
        with open('search_index_v2.json', 'r') as f:
            search_index = json.load(f)
    except:
        search_index = None 

    # Load Weather LUT
    try:
        weather_db = pd.read_csv('weather_lut_v1.csv')
        weather_db = weather_db.set_index(['ORIGIN', 'Month'])
    except:
        weather_db = None
        
    return df, search_index, weather_db

@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model('disruption_model_v5.json')
    
    with open('model_features_v5.json') as f: feats = json.load(f)
    with open('class_mapping_v5.json') as f: cmap = {int(k):v for k,v in json.load(f).items()}
    with open('model_metrics_v5.json') as f: metrics = json.load(f)
    with open('confusion_matrix_v5.json') as f: cm = json.load(f)
    with open('feature_importance_v5.json') as f: imp = json.load(f)
    
    return model, feats, cmap, metrics, cm, imp

try:
    df, search_index, airport_stats = load_data()
    model, model_features, class_map, metrics, cm_data, feature_imp = load_artifacts()
    if search_index is None: st.error("Missing search_index_v2.json")
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_weather(code):
    if code not in AIRPORT_COORDS: 
        print(f"⚠️ Debug: {code} not in coordinate list.")
        return None
    
    lat, lon = AIRPORT_COORDS[code]
    try:
        # Added a timeout to prevent hanging
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation,rain,snowfall,wind_speed_10m&wind_speed_unit=kmh"
        response = requests.get(url, timeout=5)
        
        # Check if the API actually responded with 200 OK
        if response.status_code != 200:
            print(f"⚠️ Debug: API returned status {response.status_code} for {code}")
            return None
            
        d = response.json()['current']
        return {
            'temp': d['temperature_2m'], 
            'wspd': d['wind_speed_10m'], 
            'prcp': d['precipitation'], 
            'is_snowing': 1 if (d['snowfall']>0 or (d['temperature_2m']<0 and d['precipitation']>0)) else 0
        }
    except Exception as e:
        # This will print the EXACT error to your terminal/Streamlit logs
        print(f"❌ Error fetching weather for {code}: {e}")
        return None

def infer_driver(pred, w, cong, route_risk=0):
    if pred <= 1: return ""
    if w['is_snowing']: return "❄️ Snow/Ice"
    if w['wspd'] > 40: return "💨 High Winds"
    if w['prcp'] > 5: return "🌧️ Heavy Rain"
    if cong > 45: return "🚦 Congestion"
    if route_risk > 0.6: return "⚠️ Route Fragility"
    return "✈️ Carrier Ops"

def estimate_cost(pred_class, cost_cancel=25000, cost_sev=9000, cost_mod=3000):
    if pred_class == 3: return cost_cancel 
    if pred_class == 2: return cost_sev  
    if pred_class == 1: return cost_mod  
    return 0

def make_prediction_vector(inputs, schema):
    df_in = pd.DataFrame([inputs])
    df_in['Hour_Sin'] = np.sin(2 * np.pi * df_in['Hour'] / 24)
    df_in['Hour_Cos'] = np.cos(2 * np.pi * df_in['Hour'] / 24)
    df_in['Month_Sin'] = np.sin(2 * np.pi * df_in['Month'] / 12)
    df_in['Month_Cos'] = np.cos(2 * np.pi * df_in['Month'] / 12)
    df_in['Is_Holiday'] = 0 
    encoded = pd.get_dummies(df_in)
    return encoded.reindex(columns=schema, fill_value=0)

def get_weather_bounds(airport, month, stats_df):
    try:
        s = stats_df.loc[(airport, month)]
        return (float(s['temp_min']), float(s['temp_max']), float(s['temp_mean']), float(s['wspd_max']), float(s['wspd_mean']), float(s['prcp_max']), (s['Is_Snowing_max']==1))
    except:
        return -20.0, 40.0, 15.0, 60.0, 15.0, 50.0, True

def detect_outliers(series):
    if len(series) < 10: return 0
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
    return outliers

def get_smart_options_fast(idx, c_al, c_org, c_dest):
    valid_al_set = set(idx["ALL_A"])
    if c_org: valid_al_set &= set(idx["O_A"].get(c_org, []))
    if c_dest: valid_al_set &= set(idx["D_A"].get(c_dest, []))
    
    valid_org_set = set(idx["ALL_O"])
    if c_al: valid_org_set &= set(idx["A_O"].get(c_al, []))
    if c_dest: valid_org_set &= set(idx["D_O"].get(c_dest, []))

    valid_dest_set = set(idx["ALL_D"])
    if c_al: valid_dest_set &= set(idx["A_D"].get(c_al, []))
    if c_org: valid_dest_set &= set(idx["O_D"].get(c_org, []))

    return sorted(list(valid_al_set)), sorted(list(valid_org_set)), sorted(list(valid_dest_set))

# ==========================================
# 4. NAVIGATION & LAYOUT
# ==========================================
st.title("✈️ Flight Operations Command Center")

# Top Level Navigation (Replaces Tabs)
selected_module = st.radio(
    "Go to Module:", 
    ["📡 Hub Monitor", "🧪 Simulation Lab", "🛑 Manual Risk", "📊 Analytics", "🧠 Model Insights"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# ==========================================
# 5. MODULE: LIVE MONITOR
# ==========================================
if selected_module == "📡 Hub Monitor":
    st.subheader("Real Time Hub Monitor")
    c1, c2, c3 = st.columns([1,1,2])
    with c1: hub = st.selectbox("Select Hub", sorted(AIRPORT_COORDS.keys()))
    
    local_now = datetime.now(pytz.utc).astimezone(pytz.timezone(AIRPORT_TIMEZONES[hub]))
    with c2: st.info(f"✈️ {hub}: {local_now.strftime('%H:%M')}\n\n🕒 You: {datetime.now().strftime('%H:%M')}")
    
    w = fetch_live_weather(hub)
    if w:
        with c3: 
            m1, m2, m3 = st.columns(3)
            m1.metric("Temp", f"{w['temp']}°C")
            m2.metric("Wind", f"{w['wspd']} km/h")
            m3.metric("Rain", f"{w['prcp']} mm")
            
        hhmm = local_now.hour * 100 + local_now.minute
        
        raw_sched = df[
            (df['ORIGIN']==hub) & 
            (df['Month']==local_now.month) & 
            (df['CRS_DEP_TIME']>=hhmm) &
            (df['DEST'].isin(AIRPORT_COORDS.keys()))
        ].sort_values('CRS_DEP_TIME')
        
        if raw_sched.empty:
            raw_sched = df[
                (df['ORIGIN']==hub) & 
                (df['Month']==local_now.month) & 
                (df['CRS_DEP_TIME']>=0) &
                (df['DEST'].isin(AIRPORT_COORDS.keys()))
            ].sort_values('CRS_DEP_TIME')

        sched = raw_sched.drop_duplicates(subset=['AIRLINE', 'FL_NUMBER']).head(10)

        preds = []
        arc_data = [] 
        for _, r in sched.iterrows():
            inp = {'Month': local_now.month, 'Hour': r['Hour'], 'Route_Risk_Score': r['Route_Risk_Score'], 'Is_Pandemic': 0, 'temp': w['temp'], 'wspd': w['wspd'], 'prcp': w['prcp'], 'Is_Snowing': w['is_snowing'], 'DISTANCE': r['DISTANCE'], 'Congestion': r['Congestion'], 'AIRLINE': r['AIRLINE'], 'ORIGIN': r['ORIGIN']}
            vec = make_prediction_vector(inp, model_features)
            cls = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][3]*100
            t_str = f"{int(r['CRS_DEP_TIME']):04d}"
            preds.append({"Date": local_now.strftime("%Y-%m-%d"), "Time": f"{t_str[:2]}:{t_str[2:]}", "Flight": f"{r['AIRLINE']} {r['FL_NUMBER']}", "Dest": r['DEST'], "Risk": class_map[cls], "Cancel Prob": f"{prob:.1f}%", "Likely Driver": infer_driver(cls, w, r['Congestion'], r['Route_Risk_Score'])})
            
            if r['DEST'] in AIRPORT_COORDS:
                arc_data.append({"source": [AIRPORT_COORDS[hub][1], AIRPORT_COORDS[hub][0]], "target": [AIRPORT_COORDS[r['DEST']][1], AIRPORT_COORDS[r['DEST']][0]], "name": f"{hub} -> {r['DEST']}"})
        
        st.subheader(f"Upcoming Departures from {hub}")
        st.dataframe(pd.DataFrame(preds), use_container_width=True)
        
        st.markdown("### 📍 Active Network Connectivity")
        all_airports = pd.DataFrame([{"name": k, "lon": v[1], "lat": v[0]} for k, v in AIRPORT_COORDS.items() if k != hub])
        layer_all = pdk.Layer("ScatterplotLayer", all_airports, get_position=["lon", "lat"], get_color=[200, 200, 200, 200], get_radius=50000, radius_min_pixels=3, pickable=True)
        hub_df = pd.DataFrame([{"name": hub, "lon": AIRPORT_COORDS[hub][1], "lat": AIRPORT_COORDS[hub][0]}])
        layer_hub = pdk.Layer("ScatterplotLayer", hub_df, get_position=["lon", "lat"], get_color=[255, 0, 0, 200], get_radius=100000, radius_min_pixels=10, pickable=True)
        layers = [layer_all, layer_hub]
        if arc_data:
            layer_arcs = pdk.Layer("ArcLayer", data=arc_data, get_source_position="source", get_target_position="target", get_source_color=[0, 255, 0, 160], get_target_color=[0, 255, 0, 160], get_width=4, get_height=0.5, pickable=True)
            layers.append(layer_arcs)

        view = pdk.ViewState(latitude=AIRPORT_COORDS[hub][0], longitude=AIRPORT_COORDS[hub][1], zoom=3, pitch=50)
        st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view, tooltip={"text": "{name}"}))
    else:
        st.error("Weather Service Offline.")

# ==========================================
# 6. MODULE: SIMULATION LAB
# ==========================================
elif selected_module == "🧪 Simulation Lab":
    st.header("🧪 Strategic Simulation Lab")
    
    if 'sim_al' not in st.session_state: st.session_state.sim_al = None
    if 'sim_org' not in st.session_state: st.session_state.sim_org = None
    if 'sim_dest' not in st.session_state: st.session_state.sim_dest = None
    if 'sim_results' not in st.session_state: st.session_state.sim_results = None

    def reset_sim():
        st.session_state.sim_al = None
        st.session_state.sim_org = None
        st.session_state.sim_dest = None
        st.session_state.sim_results = None 
        
    if st.button("🔄 Reset Route Selection", key="sim_reset", on_click=reset_sim):
        pass

    curr_al = st.session_state.sim_al
    curr_org = st.session_state.sim_org
    curr_dest = st.session_state.sim_dest
    
    # O(1) Lookup
    valid_als, valid_orgs, valid_dests = get_smart_options_fast(search_index, curr_al, curr_org, curr_dest)

    c1, c2, c3 = st.columns(3)
    with c1:
        idx_al = valid_als.index(curr_al) if curr_al in valid_als else 0
        new_al = st.selectbox("Airline", ["Select..."] + valid_als, index=idx_al + 1 if curr_al in valid_als else 0, key='box_sim_al')
        if new_al != "Select...": st.session_state.sim_al = new_al
        elif new_al == "Select...": st.session_state.sim_al = None

    with c2:
        idx_org = valid_orgs.index(curr_org) if curr_org in valid_orgs else 0
        new_org = st.selectbox("Origin", ["Select..."] + valid_orgs, index=idx_org + 1 if curr_org in valid_orgs else 0, key='box_sim_org')
        if new_org != "Select...": st.session_state.sim_org = new_org
        elif new_org == "Select...": st.session_state.sim_org = None

    with c3:
        idx_dest = valid_dests.index(curr_dest) if curr_dest in valid_dests else 0
        new_dest = st.selectbox("Destination", ["Select..."] + valid_dests, index=idx_dest + 1 if curr_dest in valid_dests else 0, key='box_sim_dest')
        if new_dest != "Select...": st.session_state.sim_dest = new_dest
        elif new_dest == "Select...": st.session_state.sim_dest = None

    if (new_al != "Select..." and new_al != curr_al) or (new_org != "Select..." and new_org != curr_org) or (new_dest != "Select..." and new_dest != curr_dest):
        st.rerun()

    route_valid = (st.session_state.sim_al is not None) and (st.session_state.sim_org is not None) and (st.session_state.sim_dest is not None)
    
    if route_valid:
        st.success(f"✅ Route Verified: {st.session_state.sim_al} | {st.session_state.sim_org} ➡️ {st.session_state.sim_dest}")
    else:
        st.info("ℹ️ Select all three parameters to enable simulation.")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1: sim_month = st.selectbox("Month", range(1,13), index=11, key='sim_mo')
    
    t_min, t_max, t_mean, w_max, w_mean, r_max, can_snow = (-10.0, 30.0, 15.0, 50.0, 15.0, 50.0, True)
    if st.session_state.sim_org:
        t_min, t_max, t_mean, w_max, w_mean, r_max, can_snow = get_weather_bounds(st.session_state.sim_org, sim_month, airport_stats)
    
    with col_w2: 
        sim_temp = st.slider(f"Base Temp ({t_min}° to {t_max}°)", min_value=float(t_min), max_value=float(t_max), value=float(t_mean), step=1.0, key='sim_t')
    with col_w3: pass 

    if route_valid:
        r_mask = (df['AIRLINE']==st.session_state.sim_al) & (df['ORIGIN']==st.session_state.sim_org) & (df['DEST']==st.session_state.sim_dest)
        if r_mask.sum() > 0:
            base_risk = df[r_mask]['Route_Risk_Score'].mean()
            base_dist = df[r_mask]['DISTANCE'].median()
        else:
            base_risk = df['Route_Risk_Score'].mean()
            base_dist = df['DISTANCE'].median()

        st.divider()
        col_fin1, col_fin2 = st.columns(2)
        with col_fin1: sim_cost_cancel = st.number_input("Est. Cost per Cancel ($)", value=25000, step=1000, key="sim_c_cancel")
        with col_fin2: sim_cost_delay = st.number_input("Est. Cost per Severe Delay ($)", value=9000, step=500, key="sim_c_delay")

        if st.button("Run Schedule Optimization", key="btn_sim_sweep"):
            hours = list(range(24))
            probs = []
            costs = []
            for h in hours:
                c_mask = (df['ORIGIN']==st.session_state.sim_org) & (df['Hour']==h)
                cong = df[c_mask]['Congestion'].mean() if c_mask.sum()>0 else 10
                inp = {'Month': sim_month, 'Hour': h, 'Route_Risk_Score': base_risk, 'Is_Pandemic': 0, 'temp': sim_temp, 'wspd': 15, 'prcp': 0, 'Is_Snowing': 0, 'DISTANCE': base_dist, 'Congestion': cong, 'AIRLINE': st.session_state.sim_al, 'ORIGIN': st.session_state.sim_org}
                vec = make_prediction_vector(inp, model_features)
                probs.append(model.predict_proba(vec)[0][3] * 100)
                costs.append(estimate_cost(model.predict(vec)[0], cost_cancel=sim_cost_cancel, cost_sev=sim_cost_delay))
                
            fig_sweep = px.line(x=hours, y=probs, markers=True, title=f"Risk Profile: {st.session_state.sim_org} -> {st.session_state.sim_dest}", labels={'x':'Hour','y':'Cancel %'})
            fig_sweep.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1, annotation_text="Safe")
            fig_sweep.add_hrect(y0=50, y1=100, fillcolor="red", opacity=0.1, annotation_text="Critical")
            
            export_df = pd.DataFrame({'Hour': hours, 'Cancel_Prob': probs, 'Est_Cost': costs})
            best_hour = np.argmin(probs)
            
            st.session_state.sim_results = {'fig': fig_sweep, 'df': export_df, 'best_hour': best_hour}
        
        if st.session_state.sim_results:
            st.plotly_chart(st.session_state.sim_results['fig'], use_container_width=True)
            st.download_button("Download Data", st.session_state.sim_results['df'].to_csv(index=False).encode('utf-8'), "sim_data.csv", "text/csv")
            st.success(f"Recommendation: Depart at {st.session_state.sim_results['best_hour']}:00 for lowest risk.")

        st.subheader("2. Weather Sensitivity")
        stress_var = st.selectbox("Stress Variable", ["Wind (km/h)", "Rain (mm)"], key="sim_stress")
        if st.button("Run Stress Test", key="btn_sim_stress"):
            x_vals = range(0, 100, 5)
            y_vals = []
            cong = df[(df['ORIGIN']==st.session_state.sim_org) & (df['Hour']==17)]['Congestion'].mean()
            for val in x_vals:
                w_spd = val if "Wind" in stress_var else 15
                p_rcp = val if "Rain" in stress_var else 0
                snow = 1 if (p_rcp > 0 and sim_temp < 0) else 0
                if "Rain" in stress_var and val > r_max: break 
                inp = {'Month': sim_month, 'Hour': 17, 'Route_Risk_Score': base_risk, 'Is_Pandemic': 0, 'temp': sim_temp, 'wspd': w_spd, 'prcp': p_rcp, 'Is_Snowing': snow, 'DISTANCE': base_dist, 'Congestion': cong, 'AIRLINE': st.session_state.sim_al, 'ORIGIN': st.session_state.sim_org}
                vec = make_prediction_vector(inp, model_features)
                y_vals.append(model.predict_proba(vec)[0][3] * 100)
            st.plotly_chart(px.area(x=x_vals[:len(y_vals)], y=y_vals, title=f"Risk vs {stress_var}"), use_container_width=True)

# ==========================================
# 7. MODULE: MANUAL RISK
# ==========================================
elif selected_module == "🛑 Manual Risk":
    st.header("🛑 Manual Risk Assessment")
    
    if 'man_al' not in st.session_state: st.session_state.man_al = None
    if 'man_org' not in st.session_state: st.session_state.man_org = None
    if 'man_dest' not in st.session_state: st.session_state.man_dest = None
    if 'man_results' not in st.session_state: st.session_state.man_results = None

    def reset_man():
        st.session_state.man_al = None
        st.session_state.man_org = None
        st.session_state.man_dest = None
        st.session_state.man_results = None
        
    if st.button("🔄 Reset Manual Selection", key="man_reset", on_click=reset_man):
        pass

    m_curr_al = st.session_state.man_al
    m_curr_org = st.session_state.man_org
    m_curr_dest = st.session_state.man_dest
    
    m_valid_als, m_valid_orgs, m_valid_dests = get_smart_options_fast(search_index, m_curr_al, m_curr_org, m_curr_dest)
    
    st.subheader("1. Route Selection")
    mc1, mc2, mc3 = st.columns(3)
    
    with mc1:
        m_idx_al = m_valid_als.index(m_curr_al) if m_curr_al in m_valid_als else 0
        man_al_sel = st.selectbox("Airline", ["Select..."] + m_valid_als, index=m_idx_al + 1 if m_curr_al in m_valid_als else 0, key='box_man_al')
    with mc2:
        m_idx_org = m_valid_orgs.index(m_curr_org) if m_curr_org in m_valid_orgs else 0
        man_org_sel = st.selectbox("Origin", ["Select..."] + m_valid_orgs, index=m_idx_org + 1 if m_curr_org in m_valid_orgs else 0, key='box_man_org')
    with mc3:
        m_idx_dest = m_valid_dests.index(m_curr_dest) if m_curr_dest in m_valid_dests else 0
        man_dest_sel = st.selectbox("Destination", ["Select..."] + m_valid_dests, index=m_idx_dest + 1 if m_curr_dest in m_valid_dests else 0, key='box_man_dest')

    if man_al_sel != "Select..." and man_al_sel != m_curr_al:
        st.session_state.man_al = man_al_sel
        st.rerun()
    if man_org_sel != "Select..." and man_org_sel != m_curr_org:
        st.session_state.man_org = man_org_sel
        st.rerun()
    if man_dest_sel != "Select..." and man_dest_sel != m_curr_dest:
        st.session_state.man_dest = man_dest_sel
        st.rerun()
        
    with st.form("manual_risk_form"):
        col_input, col_context = st.columns([2, 1])
        with col_input:
            st.subheader("2. Flight & Weather Details")
            c2_1, c2_2 = st.columns(2)
            with c2_1:
                man_mo = st.selectbox("Month", range(1,13), index=11, key='man_mo')
                man_hr = st.slider("Departure Hour", 0, 23, 17, key='man_hr')
                
            mt_min, mt_max, mt_mean, mw_max, mw_mean, mr_max, m_snow = (-10.0, 40.0, 15.0, 50.0, 15.0, 50.0, True)
            if man_org_sel != "Select...":
                mt_min, mt_max, mt_mean, mw_max, mw_mean, mr_max, m_snow = get_weather_bounds(man_org_sel, man_mo, airport_stats)
            
            c3, c4 = st.columns(2)
            with c3:
                man_temp = st.slider(f"Temperature (°C) [{mt_min} to {mt_max}]", min_value=float(mt_min), max_value=float(mt_max), value=float(mt_mean), step=1.0, key='man_t')
                man_wind = st.slider(f"Wind Speed (km/h) [Max: {mw_max}]", 0.0, float(mw_max), float(mw_mean), step=1.0, key='man_w')
            with c4:
                man_is_snow = st.checkbox("Snowing?", False, disabled=not m_snow, help="Disabled if snow is historically impossible.", key='man_s')
                man_rain = 5.0 if man_is_snow else st.slider(f"Rainfall (mm) [Max: {mr_max:.1f}]", 0.0, float(mr_max), 0.0, step=0.1, key='man_r')

        with col_context:
            st.markdown("### 📜 Route Status")
            if man_al_sel != "Select..." and man_org_sel != "Select..." and man_dest_sel != "Select...":
                 st.success("Route Locked.")
            else:
                 st.warning("Pending Selection.")

        st.divider()
        c_cost1, c_cost2 = st.columns(2)
        with c_cost1: man_cost_cancel = st.number_input("Est. Cost per Cancel ($)", value=25000, step=1000, key="man_c_cancel")
        with c_cost2: man_cost_delay = st.number_input("Est. Cost per Severe Delay ($)", value=9000, step=500, key="man_c_delay")

        man_submitted = st.form_submit_button("Assess Flight Risk")

    if man_submitted:
        if "Select..." in [man_al_sel, man_org_sel, man_dest_sel]:
            st.error("Please complete the Route Selection (Step 1) before assessing.")
        else:
            r_mask = (df['AIRLINE']==man_al_sel) & (df['ORIGIN']==man_org_sel) & (df['DEST']==man_dest_sel)
            risk_score = df[r_mask]['Route_Risk_Score'].mean() if r_mask.sum() > 0 else df['Route_Risk_Score'].mean()
            c_mask = (df['ORIGIN']==man_org_sel) & (df['Hour']==man_hr)
            cong_score = df[c_mask]['Congestion'].mean() if c_mask.sum() > 0 else df['Congestion'].mean()
            dist_score = df[r_mask]['DISTANCE'].median() if r_mask.sum() > 0 else df['DISTANCE'].median()
            
            inp = {'Month': man_mo, 'Hour': man_hr, 'Route_Risk_Score': risk_score, 'Is_Pandemic': 0, 'temp': man_temp, 'wspd': man_wind, 'prcp': man_rain, 'Is_Snowing': int(man_is_snow), 'DISTANCE': dist_score, 'Congestion': cong_score, 'AIRLINE': man_al_sel, 'ORIGIN': man_org_sel}
            vec = make_prediction_vector(inp, model_features)
            pred_cls = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
            cancel_prob = probs[3] * 100
            delay_prob = (probs[1] + probs[2]) * 100 
            
            est_loss = estimate_cost(pred_cls, cost_cancel=man_cost_cancel, cost_sev=man_cost_delay)
            driver = infer_driver(pred_cls, {'wspd':man_wind, 'is_snowing':int(man_is_snow), 'prcp':man_rain}, cong_score, risk_score)
            
            st.session_state.man_results = {'pred_cls': pred_cls, 'est_loss': est_loss, 'cancel_prob': cancel_prob, 'delay_prob': delay_prob, 'driver': driver}

    if st.session_state.man_results:
        res = st.session_state.man_results
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("Prediction", class_map[res['pred_cls']].upper())
        with r2: st.metric("Est. Cost Impact", f"${res['est_loss']:,}", delta_color="inverse")
        with r3: st.metric("Cancellation Probability", f"{res['cancel_prob']:.1f}%")
        with r4: st.metric("Delay Prob (Mod+Sev)", f"{res['delay_prob']:.1f}%")
            
        if res['driver']: st.warning(f"**Primary Driver:** {res['driver']}")

        g1, g2 = st.columns(2)
        with g1:
            fig_gauge1 = go.Figure(go.Indicator(
                mode = "gauge+number", value = res['cancel_prob'],
                domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Cancel Risk"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "darkred"},
                         'steps': [{'range': [0, 20], 'color': "lightgreen"}, {'range': [50, 100], 'color': "pink"}]}
            ))
            fig_gauge1.update_layout(height=200, margin={'t':0,'b':0,'l':0,'r':0})
            st.plotly_chart(fig_gauge1, use_container_width=True)
            
        with g2:
            fig_gauge2 = go.Figure(go.Indicator(
                mode = "gauge+number", value = res['delay_prob'],
                domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Delay Risk (Mod+Sev)"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "orange"},
                         'steps': [{'range': [0, 30], 'color': "lightgreen"}, {'range': [70, 100], 'color': "bisque"}]}
            ))
            fig_gauge2.update_layout(height=200, margin={'t':0,'b':0,'l':0,'r':0})
            st.plotly_chart(fig_gauge2, use_container_width=True)

# ==========================================
# 8. MODULE: ANALYTICS (HEAVY LOAD)
# ==========================================
elif selected_module == "📊 Analytics":
    st.header("Strategic Disruption Analytics")
    
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        sel_air = st.multiselect("Filter Airport", df['ORIGIN'].unique(), default=['JFK', 'ORD', 'ATL'])
    with col_ctrl2:
        select_all = st.checkbox("Select All Airports (National View)")
    
    if select_all:
        fdf = df.copy() 
    else:
        fdf = df[df['ORIGIN'].isin(sel_air)]
        
    st.divider()
    
    # 1. Financial Impact
    st.subheader("1. Financial Impact Estimation")
    min_date = fdf['FL_DATE'].min()
    max_date = fdf['FL_DATE'].max()
    date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    fdf_filtered = fdf[(fdf['FL_DATE'] >= date_range[0]) & (fdf['FL_DATE'] <= date_range[1])]
    
    col_fin_in1, col_fin_in2 = st.columns(2)
    with col_fin_in1: cost_per_cancel = st.slider("Cost per Cancellation ($)", 10000, 100000, 25000, step=5000)
    with col_fin_in2: cost_per_min = st.slider("Direct Operating Cost ($/min)", 50, 500, 100, step=10)
        
    fdf_filtered['Est_Cost'] = 0
    fdf_filtered.loc[fdf_filtered['Disruption_Severity']==3, 'Est_Cost'] = cost_per_cancel
    fdf_filtered.loc[fdf_filtered['Disruption_Severity']==2, 'Est_Cost'] = 90 * cost_per_min
    fdf_filtered.loc[fdf_filtered['Disruption_Severity']==1, 'Est_Cost'] = 30 * cost_per_min
    
    total_loss = fdf_filtered['Est_Cost'].sum()
    c1, c2 = st.columns([1, 3])
    with c1: st.metric("Total Est. Disruption Cost", f"${total_loss/1_000_000:.1f}M")
    with c2:
        daily_cost = fdf_filtered.groupby('FL_DATE')['Est_Cost'].sum().reset_index()
        st.plotly_chart(px.line(daily_cost, x='FL_DATE', y='Est_Cost', title="Daily Financial Impact Trend"), use_container_width=True)

    # 2. Operational Health
    st.subheader("2. Operational Health Overview")
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        otp_rate = (fdf_filtered['Disruption_Severity'] == 0).mean() * 100
        fig_otp = go.Figure(go.Indicator(mode = "gauge+number", value = otp_rate, title = {'text': "On-Time Performance (OTP %)"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "green"}, 'steps': [{'range': [0, 60], 'color': "red"}, {'range': [60, 80], 'color': "yellow"}]}))
        st.plotly_chart(fig_otp, use_container_width=True)
    with r2_c2:
        delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
        pres = [c for c in delay_cols if c in fdf_filtered.columns]
        if pres:
            delay_sums = fdf_filtered[pres].sum().reset_index()
            delay_sums.columns = ['Cause', 'Minutes']
            st.plotly_chart(px.pie(delay_sums, values='Minutes', names='Cause', title="Delay Attribution (Percentage)", hole=0.4), use_container_width=True)

    # 3. Airline Performance
    st.subheader("3. Airline Performance Leaderboard")
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        avg_delay = fdf_filtered.groupby('AIRLINE')['ARR_DELAY'].mean().reset_index().sort_values('ARR_DELAY')
        st.plotly_chart(px.bar(avg_delay, x='AIRLINE', y='ARR_DELAY', title="Avg Arrival Delay (Minutes) - Lower is Better"), use_container_width=True)
    with r3_c2:
        rel_stats = fdf_filtered.groupby('AIRLINE')['Disruption_Severity'].apply(lambda x: (x==0).mean()*100).reset_index(name='OTP')
        st.plotly_chart(px.bar(rel_stats.sort_values('OTP', ascending=False), x='AIRLINE', y='OTP', title="Airline Reliability (OTP %) - Higher is Better"), use_container_width=True)

    # 4. Ground Ops
    st.subheader("4. Ground Operations Analysis")
    if not fdf_filtered.empty:
        if select_all:
            fdf_filtered['Plot_Group'] = "National Aggregate"
        else:
            stats_df = fdf_filtered.groupby('ORIGIN')['TAXI_OUT'].agg(['count', detect_outliers]).reset_index()
            stats_df['Label'] = stats_df.apply(lambda x: f"{x['ORIGIN']} (n={int(x['count']):,} | {int(x['detect_outliers'])} outliers)", axis=1)
            label_map = dict(zip(stats_df['ORIGIN'], stats_df['Label']))
            fdf_filtered['Plot_Group'] = fdf_filtered['ORIGIN'].map(label_map)
    
    r4_c1, r4_c2 = st.columns(2)
    with r4_c1:
        if 'TAXI_OUT' in fdf_filtered.columns: 
            st.plotly_chart(px.box(fdf_filtered, x='Plot_Group', y='TAXI_OUT', title="Taxi-Out Times (Dep Congestion) - Min"), use_container_width=True)
    with r4_c2:
        if 'TAXI_IN' in fdf_filtered.columns: 
            st.plotly_chart(px.box(fdf_filtered, x='Plot_Group', y='TAXI_IN', title="Taxi-In Times (Arr Congestion) - Min"), use_container_width=True)

    # 5. Problem Areas
    st.subheader("5. Problem Areas (Route Analysis)")
    r5_c1, r5_c2 = st.columns(2)
    
    route_perf = fdf_filtered.groupby(['AIRLINE', 'ORIGIN', 'DEST']).agg(
        Total_Flights=('FL_DATE', 'count'),
        Cancel_Count=('Disruption_Severity', lambda x: (x==3).sum()),
        Avg_Delay=('ARR_DELAY', 'mean')
    ).reset_index()
    
    valid_routes = route_perf[route_perf['Total_Flights'] > 20]
    
    with r5_c1:
        st.markdown("**Top 10 Highest Cancellation Rates**")
        valid_routes['Cancel_Rate'] = (valid_routes['Cancel_Count'] / valid_routes['Total_Flights']) * 100
        bad_cancels = valid_routes.sort_values('Cancel_Rate', ascending=False).head(10)
        st.dataframe(bad_cancels[['AIRLINE','ORIGIN','DEST','Cancel_Rate']].style.format({'Cancel_Rate': "{:.1f}%"}), use_container_width=True, hide_index=True)
        
    with r5_c2:
        st.markdown("**Top 10 Highest Average Delays**")
        bad_delays = valid_routes.sort_values('Avg_Delay', ascending=False).head(10)
        st.dataframe(bad_delays[['AIRLINE','ORIGIN','DEST','Avg_Delay']].style.format({'Avg_Delay': "{:.1f} min"}), use_container_width=True, hide_index=True)
        
# ==========================================
# 9. MODULE: MODEL INSIGHTS
# ==========================================
elif selected_module == "🧠 Model Insights":
    st.header("🧠 Model Transparency & Live Validation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("F1 Score", f"{metrics['f1_weighted']:.2f}")
    m3.metric("Test Data", f"{metrics['test_samples']:,} flights")
    st.divider()
    st.subheader("🧪 Real-Time Model Validation")
    st.caption("Testing the model on a stratified sample (balanced mix of delays/cancellations) to validate error patterns.")
    if st.button("Run Live Validation"):
        sample = df.sample(n=500)
        inputs = sample[['Month', 'Hour', 'Route_Risk_Score', 'Is_Pandemic', 'temp', 'wspd', 'prcp', 'Is_Snowing', 'DISTANCE', 'Congestion', 'AIRLINE', 'ORIGIN']]
        inputs['Hour_Sin'] = np.sin(2 * np.pi * inputs['Hour'] / 24)
        inputs['Hour_Cos'] = np.cos(2 * np.pi * inputs['Hour'] / 24)
        inputs['Month_Sin'] = np.sin(2 * np.pi * inputs['Month'] / 12)
        inputs['Month_Cos'] = np.cos(2 * np.pi * inputs['Month'] / 12)
        inputs['Is_Holiday'] = 0
        encoded = pd.get_dummies(inputs)
        aligned = encoded.reindex(columns=model_features, fill_value=0)
        preds = model.predict(aligned)
        sample['Predicted'] = preds
        sample['Correct'] = (sample['Disruption_Severity'] == sample['Predicted'])
        c1, c2 = st.columns([1, 1]) 
        with c1:
            acc_by_al = sample.groupby('AIRLINE')['Correct'].mean().reset_index().sort_values('Correct')
            st.plotly_chart(px.bar(acc_by_al, x='Correct', y='AIRLINE', orientation='h', title="Live Accuracy by Airline"), use_container_width=True)
        with c2:
            st.markdown("##### 📋 Class Performance")
            from sklearn.metrics import classification_report
            report_dict = classification_report(sample['Disruption_Severity'], preds, output_dict=True, zero_division=0)
            accuracy_val = report_dict.pop('accuracy')
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.drop(['macro avg', 'weighted avg'], errors='ignore')
            index_map = {'0': 'On-Time', '1': 'Moderate', '2': 'Severe', '3': 'Cancelled'}
            report_df = report_df.rename(index=index_map)
            report_df.columns = ['Precision', 'Recall', 'F1 / Accuracy', 'Support']
            total_support = report_df['Support'].sum()
            report_df.loc['Total Accuracy'] = [np.nan, np.nan, accuracy_val, total_support]
            def style_footer(row):
                if row.name == 'Total Accuracy':
                    return ['font-weight: bold; border-top: 2px solid #444; background-color: #f9f9f9'] * len(row)
                return [''] * len(row)
            styled_df = report_df.style\
                .format("{:.2f}", subset=['Precision', 'Recall', 'F1 / Accuracy'], na_rep="")\
                .format("{:.0f}", subset=['Support'])\
                .background_gradient(cmap="Blues", subset=['Recall'], vmin=0, vmax=1)\
                .apply(style_footer, axis=1)  
            st.dataframe(styled_df, use_container_width=True)

# Force Memory Cleanup at end of script run
gc.collect()


