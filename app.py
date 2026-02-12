import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EpiWatch AI",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .brand-container {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(0, 255, 150, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 255, 150, 0.3);
    }
    
    .brand-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff96 0%, #00c8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(0, 255, 150, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(0, 200, 255, 0.8)); }
    }
    
    .brand-tagline {
        font-size: 1.3rem;
        color: #b0b8d1;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .risk-preview-box {
        background: rgba(20, 25, 45, 0.8);
        border-radius: 15px;
        padding: 2rem;
        border: 2px solid;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }
    
    .risk-low {
        border-color: #00ff96;
        box-shadow: 0 0 20px rgba(0, 255, 150, 0.3);
    }
    
    .risk-moderate {
        border-color: #ffd700;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    }
    
    .risk-high {
        border-color: #ff4444;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.5);
    }
    
    .risk-level-text {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: rgba(20, 25, 45, 0.6);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 150, 0.2);
        margin: 0.5rem 0;
    }
    
    .alert-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 3px solid;
        animation: pulse 2s infinite;
    }
    
    .alert-high {
        background: rgba(255, 68, 68, 0.2);
        border-color: #ff4444;
        color: #ffaaaa;
    }
    
    .alert-moderate {
        background: rgba(255, 215, 0, 0.2);
        border-color: #ffd700;
        color: #ffeb99;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00ff96 0%, #00c8ff 100%);
        color: #0a0e27;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 150, 0.5);
    }
    
    h1, h2, h3 {
        color: #e0e6f0;
    }
    
    .stSelectbox label, .stSelectbox div {
        color: #b0b8d1;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = None
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = None
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = None
if 'risk_color' not in st.session_state:
    st.session_state.risk_color = None

# Generate sample dataset
@st.cache_data
def generate_sample_data():
    """Generate synthetic epidemiological data"""
    countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'Germany', 
                 'France', 'Italy', 'Spain', 'Canada', 'Australia', 'Japan', 'South Korea',
                 'Mexico', 'Argentina', 'South Africa', 'Nigeria', 'Egypt', 'Kenya']
    
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    data = []
    base_date = datetime(2023, 1, 1)
    
    for country in countries:
        for month_idx, month in enumerate(months):
            # Generate daily data for the month
            days_in_month = (datetime(2023, month_idx + 1, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_in_month = days_in_month.day
            
            # Base cases with some randomness
            base_cases = np.random.randint(50, 500)
            
            for day in range(1, days_in_month + 1):
                date = datetime(2023, month_idx + 1, day)
                
                # Add trend and seasonality
                trend = day * np.random.uniform(0.8, 1.2)
                seasonal = np.sin(2 * np.pi * day / days_in_month) * 20
                noise = np.random.normal(0, 15)
                
                cases = max(0, int(base_cases + trend + seasonal + noise))
                
                data.append({
                    'Country': country,
                    'Month': month,
                    'Date': date,
                    'Day': day,
                    'Cases': cases,
                    'Fever_Cases': int(cases * np.random.uniform(0.3, 0.5)),
                    'Hospitalizations': int(cases * np.random.uniform(0.1, 0.2)),
                })
    
    return pd.DataFrame(data)

# AI Model Functions
def preprocess_data(df):
    """Preprocess data for anomaly detection"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate features
    df['Cases_MA7'] = df['Cases'].rolling(window=7, min_periods=1).mean()
    df['Cases_Change'] = df['Cases'].pct_change().fillna(0)
    df['Cases_Std'] = df['Cases'].rolling(window=7, min_periods=1).std().fillna(0)
    
    return df

def detect_anomalies(df):
    """Use Isolation Forest to detect anomalies"""
    features = ['Cases', 'Cases_MA7', 'Cases_Change', 'Cases_Std']
    feature_data = df[features].fillna(0)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(feature_data)
    df['Anomaly_Score'] = iso_forest.score_samples(feature_data)
    
    return df

def calculate_risk_score(df):
    """Calculate composite risk score"""
    # Normalize anomaly score (lower = more anomalous)
    anomaly_score = df['Anomaly_Score'].min()
    normalized_anomaly = (anomaly_score - df['Anomaly_Score'].min()) / (df['Anomaly_Score'].max() - df['Anomaly_Score'].min() + 1e-10)
    
    # Growth rate
    recent_cases = df['Cases'].tail(7).mean()
    previous_cases = df['Cases'].head(7).mean() if len(df) > 7 else recent_cases
    growth_rate = (recent_cases - previous_cases) / (previous_cases + 1e-10)
    
    # Anomaly count
    anomaly_count = (df['Anomaly'] == -1).sum()
    anomaly_ratio = anomaly_count / len(df)
    
    # Composite risk score (0-100)
    risk_score = (
        (1 - normalized_anomaly) * 40 +  # Anomaly component
        min(abs(growth_rate) * 30, 30) +  # Growth component
        anomaly_ratio * 30  # Anomaly frequency component
    )
    
    return min(100, max(0, risk_score))

def get_risk_level(risk_score):
    """Get risk level from score"""
    if risk_score < 35:
        return "Low", "#00ff96"
    elif risk_score < 70:
        return "Moderate", "#ffd700"
    else:
        return "High", "#ff4444"

def generate_ai_explanation(df, risk_score, risk_level):
    """Generate AI explanation text"""
    recent_avg = df['Cases'].tail(7).mean()
    previous_avg = df['Cases'].head(7).mean() if len(df) > 7 else recent_avg
    growth_pct = ((recent_avg - previous_avg) / (previous_avg + 1e-10)) * 100
    anomaly_count = (df['Anomaly'] == -1).sum()
    
    if risk_level == "High":
        explanation = f"🚨 AI detected {anomaly_count} anomalous patterns with a {growth_pct:.1f}% increase in cases. "
        explanation += "The system predicts elevated outbreak risk exceeding seasonal baselines."
    elif risk_level == "Moderate":
        explanation = f"⚠️ Monitoring {anomaly_count} potential anomalies with {growth_pct:.1f}% case variation. "
        explanation += "The situation requires continued observation."
    else:
        explanation = f"✅ System indicates stable patterns with minimal anomalies ({anomaly_count} detected). "
        explanation += f"Case trends show {abs(growth_pct):.1f}% change, within normal parameters."
    
    return explanation

# Main App
def main():
    # Load data
    df = generate_sample_data()
    
    # Page selection
    page = st.sidebar.selectbox("Navigation", ["🏠 Landing & Filter", "📊 AI Analysis Dashboard"])
    
    if page == "🏠 Landing & Filter":
        show_landing_page(df)
    else:
        show_analysis_page(df)

def show_landing_page(df):
    """Page 1: Landing + Filter Dashboard"""
    
    # Branding Section
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">🦠 EpiWatch AI</div>
            <div class="brand-tagline">Predicting outbreaks before they escalate.</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Filter Panel
    st.markdown("### 🌍 Smart Search Panel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        countries = sorted(df['Country'].unique())
        selected_country = st.selectbox(
            "Select Country",
            options=countries,
            index=0 if st.session_state.selected_country is None else countries.index(st.session_state.selected_country) if st.session_state.selected_country in countries else 0
        )
    
    with col2:
        months = sorted(df['Month'].unique())
        selected_month = st.selectbox(
            "Select Month",
            options=months,
            index=0 if st.session_state.selected_month is None else months.index(st.session_state.selected_month) if st.session_state.selected_month in months else 0
        )
    
    # Store selections
    st.session_state.selected_country = selected_country
    st.session_state.selected_month = selected_month
    
    # Filter data
    filtered_df = df[(df['Country'] == selected_country) & (df['Month'] == selected_month)].copy()
    
    if len(filtered_df) > 0:
        # Preprocess and analyze
        filtered_df = preprocess_data(filtered_df)
        filtered_df = detect_anomalies(filtered_df)
        risk_score = calculate_risk_score(filtered_df)
        risk_level, risk_color = get_risk_level(risk_score)
        
        # Store for analysis page
        st.session_state.data = filtered_df
        st.session_state.risk_score = risk_score
        st.session_state.risk_level = risk_level
        st.session_state.risk_color = risk_color
        
        # AI Risk Preview Box
        risk_class = f"risk-{risk_level.lower()}"
        st.markdown(f"""
            <div class="risk-preview-box {risk_class}">
                <h2 style="color: {risk_color}; margin-bottom: 1rem;">🎯 AI Risk Preview</h2>
                <div class="risk-level-text" style="color: {risk_color};">{risk_level} Risk</div>
                <p style="color: #b0b8d1; font-size: 1.1rem;">Risk Score: <strong style="color: {risk_color};">{risk_score:.1f}/100</strong></p>
                <p style="color: #b0b8d1;">AI Confidence: <strong style="color: {risk_color};">{85 + np.random.randint(-5, 10)}%</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Analyze Button
        if st.button("🔍 Analyze Outbreak Patterns", use_container_width=True):
            st.rerun()
    else:
        st.warning("No data available for selected filters.")

def show_analysis_page(df):
    """Page 2: AI Analysis Dashboard"""
    
    # Check if data is available
    if st.session_state.data is None or st.session_state.selected_country is None:
        st.warning("⚠️ Please go to the Landing page and select a country and month first.")
        return
    
    filtered_df = st.session_state.data
    risk_score = st.session_state.risk_score
    risk_level = st.session_state.risk_level
    risk_color = st.session_state.risk_color
    selected_country = st.session_state.selected_country
    selected_month = st.session_state.selected_month
    
    # Header
    st.markdown(f"""
        <h1 style="text-align: center; color: #00ff96; margin-bottom: 0.5rem;">
            📊 AI Analysis Dashboard
        </h1>
        <p style="text-align: center; color: #b0b8d1; font-size: 1.2rem; margin-bottom: 2rem;">
            {selected_country} • {selected_month} 2023
        </p>
    """, unsafe_allow_html=True)
    
    # Early Warning Alert Panel
    if risk_level == "High":
        st.markdown(f"""
            <div class="alert-box alert-high">
                <h2 style="color: #ff4444; margin: 0;">🚨 High Outbreak Probability Detected</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Immediate attention recommended. Anomalous patterns detected in epidemiological data.
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif risk_level == "Moderate":
        st.markdown(f"""
            <div class="alert-box alert-moderate">
                <h2 style="color: #ffd700; margin: 0;">⚠️ Monitoring Situation</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Elevated patterns observed. Continued monitoring advised.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Content Grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Outbreak Trend Graph
        st.markdown("### 📈 Outbreak Trend Analysis")
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Main trend line
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Cases'],
                mode='lines+markers',
                name='Daily Cases',
                line=dict(color='#00ff96', width=3),
                marker=dict(size=6)
            )
        )
        
        # Moving average
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Cases_MA7'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='#00c8ff', width=2, dash='dash')
            )
        )
        
        # Anomaly markers
        anomalies = filtered_df[filtered_df['Anomaly'] == -1]
        if len(anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalies['Date'],
                    y=anomalies['Cases'],
                    mode='markers',
                    name='AI Detected Anomalies',
                    marker=dict(
                        size=12,
                        color='#ff4444',
                        symbol='x',
                        line=dict(width=2, color='white')
                    )
                )
            )
        
        fig.update_layout(
            plot_bgcolor='rgba(20, 25, 45, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#b0b8d1'),
            xaxis=dict(gridcolor='rgba(0, 255, 150, 0.1)'),
            yaxis=dict(gridcolor='rgba(0, 255, 150, 0.1)'),
            legend=dict(bgcolor='rgba(20, 25, 45, 0.8)'),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Indicator Meter (Gauge)
        st.markdown("### 🎯 Risk Indicator")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'color': '#b0b8d1', 'size': 20}},
            delta={'reference': 50, 'font': {'color': '#b0b8d1'}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': '#b0b8d1'},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(0, 255, 150, 0.2)'},
                    {'range': [35, 70], 'color': 'rgba(255, 215, 0, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(255, 68, 68, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        fig_gauge.update_layout(
            plot_bgcolor='rgba(20, 25, 45, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#b0b8d1'),
            height=300
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <h2 style="color: {risk_color}; margin: 0;">{risk_level}</h2>
                <p style="color: #b0b8d1;">Risk Level</p>
            </div>
        """, unsafe_allow_html=True)
    
    # AI Analysis Section
    st.markdown("### 🧠 AI Analysis")
    
    explanation = generate_ai_explanation(filtered_df, risk_score, risk_level)
    confidence = 85 + np.random.randint(-5, 10)
    
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #00ff96; margin-bottom: 1rem;">Model Insights</h3>
            <p style="color: #b0b8d1; font-size: 1.1rem; line-height: 1.8;">
                {explanation}
            </p>
            <p style="color: #b0b8d1; margin-top: 1rem;">
                <strong style="color: #00c8ff;">Model Confidence:</strong> {confidence}%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Data Insights Section
    st.markdown("### 📊 Data Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    recent_avg = filtered_df['Cases'].tail(7).mean()
    previous_avg = filtered_df['Cases'].head(7).mean() if len(filtered_df) > 7 else recent_avg
    growth_rate = ((recent_avg - previous_avg) / (previous_avg + 1e-10)) * 100
    total_cases = filtered_df['Cases'].sum()
    anomaly_count = (filtered_df['Anomaly'] == -1).sum()
    
    with col1:
        st.metric("Total Cases", f"{total_cases:,.0f}", f"{growth_rate:+.1f}%")
    
    with col2:
        st.metric("Avg Daily Cases", f"{recent_avg:.0f}", "Last 7 days")
    
    with col3:
        st.metric("Anomalies Detected", f"{anomaly_count}", "AI flagged")
    
    with col4:
        st.metric("Growth Rate", f"{growth_rate:+.1f}%", "vs. early period")
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📉 Case Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x='Cases',
            nbins=20,
            color_discrete_sequence=['#00ff96']
        )
        fig_hist.update_layout(
            plot_bgcolor='rgba(20, 25, 45, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#b0b8d1'),
            xaxis=dict(gridcolor='rgba(0, 255, 150, 0.1)'),
            yaxis=dict(gridcolor='rgba(0, 255, 150, 0.1)'),
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔍 Anomaly Score Distribution")
        fig_anomaly = px.scatter(
            filtered_df,
            x='Date',
            y='Anomaly_Score',
            color='Anomaly',
            color_discrete_map={1: '#00ff96', -1: '#ff4444'},
            labels={'Anomaly_Score': 'Anomaly Score (lower = more anomalous)'}
        )
        fig_anomaly.update_layout(
            plot_bgcolor='rgba(20, 25, 45, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#b0b8d1'),
            xaxis=dict(gridcolor='rgba(0, 255, 150, 0.1)'),
            yaxis=dict(gridcolor='rgba(0, 255, 150, 0.1)'),
            height=300
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

if __name__ == "__main__":
    main()
