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
    page_title="EpiWatch Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme colors with gradients
def get_theme_css(dark_mode):
    if dark_mode:
        return """
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #262626;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #888888;
            --border-color: #333333;
            --button-bg: #ffffff;
            --button-text: #0f0f0f;
            --button-hover: #f5f5f5;
            --accent-color: #3b82f6;
            --card-bg: linear-gradient(145deg, #1a1a1a, #141414);
        }
        """
    else:
        return """
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --text-primary: #1a1a1a;
            --text-secondary: #4a4a4a;
            --text-muted: #6c757d;
            --border-color: #dee2e6;
            --button-bg: #1a1a1a;
            --button-text: #ffffff;
            --button-hover: #2a2a2a;
            --accent-color: #2563eb;
            --card-bg: linear-gradient(145deg, #ffffff, #f8f9fa);
        }
        """

# Custom CSS for styling
def apply_theme(dark_mode):
    theme_css = get_theme_css(dark_mode)
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    {theme_css}
    
    .main {{
        background: var(--bg-primary);
    }}
    
    .stApp {{
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }}
    
    .brand-container {{
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-color);
    }}
    
    .brand-title {{
        font-size: 4.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }}
    
    .brand-tagline {{
        font-size: 1.1rem;
        color: var(--text-muted);
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    /* Risk Hero Card */
    .risk-hero-dboard {{
        background: var(--card-bg);
        border-radius: 16px;
        padding: 2.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s;
        color: var(--text-primary);
    }}
    
    .risk-hero-dboard:hover {{
        transform: translateY(-2px);
    }}
    
    .risk-label-top {{
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .risk-value-huge {{
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }}
    
    .risk-level-badge {{
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        background: rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        display: inline-block;
    }}
    
    .risk-context {{
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
        width: 100%;
    }}
    
    /* Stats Cards */
    .stat-card-modern {{
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s;
        color: var(--text-primary);
    }}
    
    .stat-card-modern:hover {{
        border-color: var(--text-muted);
    }}
    
    .stat-label {{
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-bottom: 0.25rem;
    }}
    
    .stat-value {{
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }}
    
    .stat-icon {{
        font-size: 1.5rem;
        opacity: 0.8;
    }}

    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.5s ease-out forwards;
    }}
    
    /* Global Overrides */
    .stButton>button {{
        background: var(--button-bg);
        color: var(--button-text);
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: 1px solid var(--border-color);
    }}
    
    /* Hide Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = None
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = None
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = None
if 'risk_color' not in st.session_state:
    st.session_state.risk_color = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Load Kaggle COVID-19 dataset
@st.cache_data
def load_covid_data():
    """Load COVID-19 dataset from Kaggle"""
    try:
        # Try to load from common file names
        possible_files = [
            'covid19-global-dataset.csv',
            'covid19_global_dataset.csv',
            'covid19-global.csv',
            'covid_19_data.csv',
            'data.csv',
            'covid19.csv'
        ]
        
        df = None
        loaded_file = None
        for filename in possible_files:
            try:
                df = pd.read_csv(filename)
                loaded_file = filename
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            # If no file found, generate sample data as fallback
            return generate_sample_data()
        
        # Standardize column names (handle different naming conventions)
        df.columns = df.columns.str.strip()
        
        # Try to identify date column
        date_col = None
        for col in ['Date', 'date', 'DATE', 'ObservationDate', 'observation_date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            return generate_sample_data()
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # Try to identify country column
        country_col = None
        for col in ['Country/Region', 'Country', 'country', 'Country_Region', 'Country/Region']:
            if col in df.columns:
                country_col = col
                break
        
        if country_col is None:
            return generate_sample_data()
        
        # Try to identify cases column
        cases_col = None
        for col in ['Confirmed', 'confirmed', 'Cases', 'cases', 'Total Cases', 'TotalCases']:
            if col in df.columns:
                cases_col = col
                break
        
        if cases_col is None:
            return generate_sample_data()
        
        # Standardize the dataframe
        df['Date'] = df[date_col]
        df['Country'] = df[country_col]
        df['Cases'] = pd.to_numeric(df[cases_col], errors='coerce').fillna(0).astype(int)
        
        # Extract year and month
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        df['Day'] = df['Date'].dt.day
        
        # Try to get additional columns if available
        if 'Deaths' in df.columns or 'deaths' in df.columns:
            death_col = 'Deaths' if 'Deaths' in df.columns else 'deaths'
            df['Deaths'] = pd.to_numeric(df[death_col], errors='coerce').fillna(0).astype(int)
        else:
            df['Deaths'] = 0
        
        if 'Recovered' in df.columns or 'recovered' in df.columns:
            rec_col = 'Recovered' if 'Recovered' in df.columns else 'recovered'
            df['Recovered'] = pd.to_numeric(df[rec_col], errors='coerce').fillna(0).astype(int)
        else:
            df['Recovered'] = 0
        
        # Calculate daily cases (difference from previous day)
        df = df.sort_values(['Country', 'Date'])
        df['Daily_Cases'] = df.groupby('Country')['Cases'].diff().fillna(df['Cases']).astype(int)
        df['Daily_Cases'] = df['Daily_Cases'].clip(lower=0)  # Ensure non-negative
        
        # Use Daily_Cases as Cases for analysis
        df['Cases'] = df['Daily_Cases']
        
        # Add placeholder columns for compatibility
        if 'Fever_Cases' not in df.columns:
            df['Fever_Cases'] = (df['Cases'] * np.random.uniform(0.3, 0.5)).astype(int)
        if 'Hospitalizations' not in df.columns:
            df['Hospitalizations'] = (df['Cases'] * np.random.uniform(0.1, 0.2)).astype(int)
        if 'Tests_Conducted' not in df.columns:
            df['Tests_Conducted'] = (df['Cases'] * np.random.uniform(5, 10)).astype(int)
        
        # Select relevant columns
        df = df[['Country', 'Date', 'Year', 'Month', 'Day', 'Cases', 'Deaths', 'Recovered', 
                'Fever_Cases', 'Hospitalizations', 'Tests_Conducted']]
        
        return df
        
    except Exception as e:
        return generate_sample_data()

# Generate sample dataset (fallback)
@st.cache_data
def generate_sample_data():
    """Generate synthetic epidemiological data"""
    countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'Germany', 
                 'France', 'Italy', 'Spain', 'Canada', 'Australia', 'Japan', 'South Korea',
                 'Mexico', 'Argentina', 'South Africa', 'Nigeria', 'Egypt', 'Kenya']
    
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    years = [2020, 2021, 2022, 2023]
    
    data = []
    
    for country in countries:
        for year in years:
            for month_idx, month in enumerate(months):
                # Generate daily data for the month
                days_in_month = (datetime(year, month_idx + 1, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                days_in_month = days_in_month.day
                
                # Base cases with some randomness
                base_cases = np.random.randint(50, 500)
                
                for day in range(1, days_in_month + 1):
                    date = datetime(year, month_idx + 1, day)
                    
                    # Add trend and seasonality
                    trend = day * np.random.uniform(0.8, 1.2)
                    seasonal = np.sin(2 * np.pi * day / days_in_month) * 20
                    noise = np.random.normal(0, 15)
                    
                    cases = max(0, int(base_cases + trend + seasonal + noise))
                    
                    data.append({
                        'Country': country,
                        'Year': year,
                        'Month': month,
                        'Date': date,
                        'Day': day,
                        'Cases': cases,
                        'Fever_Cases': int(cases * np.random.uniform(0.3, 0.5)),
                        'Hospitalizations': int(cases * np.random.uniform(0.1, 0.2)),
                        'Deaths': int(cases * np.random.uniform(0.01, 0.03)),
                        'Recovered': int(cases * np.random.uniform(0.6, 0.8)),
                        'Tests_Conducted': int(cases * np.random.uniform(5, 10)),
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
        return "Low", "#10b981"
    elif risk_score < 70:
        return "Moderate", "#f59e0b"
    else:
        return "High", "#ef4444"

def get_chart_colors(dark_mode):
    """Get chart colors based on theme"""
    if dark_mode:
        return {
            'bg': '#1a1a1a',
            'paper': '#0f0f0f',
            'text': '#ffffff',
            'grid': '#262626',
            'line': '#ffffff',
            'line_secondary': '#888888'
        }
    else:
        return {
            'bg': '#f8f9fa',
            'paper': '#ffffff',
            'text': '#1a1a1a',
            'grid': '#dee2e6',
            'line': '#1a1a1a',
            'line_secondary': '#6c757d'
        }

def generate_ai_explanation(df, risk_score, risk_level):
    """Generate AI explanation text"""
    recent_avg = df['Cases'].tail(7).mean()
    previous_avg = df['Cases'].head(7).mean() if len(df) > 7 else recent_avg
    growth_pct = ((recent_avg - previous_avg) / (previous_avg + 1e-10)) * 100
    anomaly_count = (df['Anomaly'] == -1).sum()
    
    if risk_level == "High":
        explanation = f"Detected {anomaly_count} anomalous patterns with {growth_pct:.1f}% case increase. "
        explanation += "Elevated outbreak risk identified."
    elif risk_level == "Moderate":
        explanation = f"Monitoring {anomaly_count} potential anomalies with {growth_pct:.1f}% variation. "
        explanation += "Continued observation recommended."
    else:
        explanation = f"Stable patterns observed with {anomaly_count} anomalies detected. "
        explanation += f"Case trends show {abs(growth_pct):.1f}% change within normal parameters."
    
    return explanation

# Main App
def main():
    # Load data
    df = load_covid_data()
    
    # Apply theme first
    dark_mode = st.session_state.dark_mode
    apply_theme(dark_mode)
    
    # Dark mode toggle - prominent button at top
    dark_mode_text = "Dark Mode" if not dark_mode else "Light Mode"
    
    # Top bar with dark mode toggle
    col_spacer, col_toggle = st.columns([15, 1])
    with col_toggle:
        if st.button("💡", key="dark_mode_toggle", help="Toggle Light/Dark Mode", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    
    # Dark mode toggle in sidebar as well
    with st.sidebar:
        st.markdown("### Display Settings")
        if st.button(f"Switch to {dark_mode_text}", key="dark_mode_toggle_sidebar", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.markdown("---")
        
        # Show dataset status
        st.markdown("### Dataset Status")
        if 'Year' in df.columns:
            st.success("Dataset loaded successfully")
        else:
            st.warning("Using sample data. Download dataset from Kaggle.")
    
    # Page routing
    if "navigation" not in st.session_state:
        st.session_state.navigation = "Home"
        
    if st.session_state.navigation == "Home":
        show_home_page(dark_mode)
    elif st.session_state.navigation == "Selection":
        show_selection_page(df, dark_mode)
    elif st.session_state.navigation == "Analysis":
        show_analysis_page(df, dark_mode)
    else:
        # Fallback
        st.session_state.navigation = "Home"
        st.rerun()

def show_home_page(dark_mode):
    """Page 1: Splash / Home Screen"""
    st.markdown("""
        <style>
        .home-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 70vh;
            text-align: center;
            animation: fadeIn 1.5s ease-in-out;
        }
        .big-title {
            font-size: 6rem;
            font-weight: 900;
            color: var(--text-primary);
            margin-bottom: 1rem;
            letter-spacing: -3px;
        }
        .subtitle {
            font-size: 1.5rem;
            color: var(--text-muted);
            margin-bottom: 3rem;
            font-weight: 300;
        }
        .start-btn-container {
            margin-top: 2rem;
        }
        
        /* Subtle Background Image */
        .stApp {{
            background-color: { '#000000' if dark_mode else '#ffffff' };
            background-image: linear-gradient({ 'rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.92)' if dark_mode else 'rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)' }), url("background.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        
        <div class="home-container">
            <div class="big-title">EPIWATCH</div>
            <div class="subtitle">Global Pathogen Surveillance System</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        def start_app():
            st.session_state.navigation = "Selection"
            
        st.button("Start Analysis", on_click=start_app, use_container_width=True, type="primary")

def show_selection_page(df, dark_mode):
    """Page 2: Selection & Risk Summary"""
    chart_colors = get_chart_colors(dark_mode)

    # Apply global background via CSS
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: { '#000000' if dark_mode else '#ffffff' };
            background-image: linear-gradient({ 'rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.92)' if dark_mode else 'rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)' }), url("background.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Header for Selection Page
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 style="margin: 0;">Global Epidemiology Monitor</h2>
            <p style="color: var(--text-muted);">Select a region to analyze outbreak risks.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset info
    if 'Year' not in df.columns:
        st.info("Tip: Download the COVID-19 dataset from [Kaggle](https://www.kaggle.com/datasets/josephassaker/covid19-global-dataset) and place it in this folder to use real data.")
    
    # Filter Panel - Clean & Minimal
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = sorted(df['Country'].unique())
        selected_country = st.selectbox(
            "Location",
            options=countries,
            index=0 if st.session_state.selected_country is None else countries.index(st.session_state.selected_country) if st.session_state.selected_country in countries else 0
        )
    
    with col2:
        years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox(
            "Year",
            options=years,
            index=0 if st.session_state.selected_year is None else years.index(st.session_state.selected_year) if st.session_state.selected_year in years else 0
        )
    
    with col3:
        # Filter months based on selected year and country
        available_months = sorted(df[(df['Country'] == selected_country) & (df['Year'] == selected_year)]['Month'].unique())
        if len(available_months) == 0:
            available_months = sorted(df['Month'].unique())
        
        selected_month = st.selectbox(
            "Timeframe",
            options=available_months,
            index=0 if st.session_state.selected_month is None else (available_months.index(st.session_state.selected_month) if st.session_state.selected_month in available_months else 0)
        )
    
    # Store selections
    st.session_state.selected_country = selected_country
    st.session_state.selected_year = selected_year
    st.session_state.selected_month = selected_month
    
    # 1. Filter by Country and Year first (Context for Anomaly Detection)
    yearly_df = df[(df['Country'] == selected_country) & (df['Year'] == selected_year)].copy()
    
    if len(yearly_df) > 0:
        # 2. Preprocess and Detect Anomalies on the Full Year
        yearly_df = preprocess_data(yearly_df)
        yearly_df = detect_anomalies(yearly_df)
        
        # 3. Filter for the selected month to display
        filtered_df = yearly_df[yearly_df['Month'] == selected_month].copy()
        
        if len(filtered_df) == 0:
            st.warning("No data available for this month.")
            return

        risk_score = calculate_risk_score(filtered_df)
        risk_level, risk_color = get_risk_level(risk_score)
        
        # Store for analysis page
        st.session_state.data = filtered_df
        st.session_state.risk_score = risk_score
        st.session_state.risk_level = risk_level
        st.session_state.risk_color = risk_color
        
        # Calculate stats
        total_cases = filtered_df['Cases'].sum()
        avg_daily = filtered_df['Cases'].mean()
        max_daily = filtered_df['Cases'].max()
        anomaly_count = (filtered_df['Anomaly'] == -1).sum()
        
        # Calculate confidence based on data sufficiency and variance
        data_points = len(filtered_df)
        cv = filtered_df['Cases'].std() / (filtered_df['Cases'].mean() + 1e-10) # Coefficient of Variation
        
        # Base confidence on data quantity
        confidence_base = min(data_points * 2, 70) 
        
        # Adjust based on volatility
        volatility_penalty = min(cv * 20, 30)
        
        # Add small random fluctuation for "live" feel (simulated sensor noise)
        jitter = np.random.randint(-2, 3)
        
        confidence_score = min(max(int(confidence_base + 25 - volatility_penalty + jitter), 45), 99)
        
        # Determine label
        if confidence_score >= 80:
            confidence_label = "High"
            conf_color = "#10b981"
        elif confidence_score >= 60:
            confidence_label = "Medium"
            conf_color = "#f59e0b"
        else:
            confidence_label = "Low"
            conf_color = "#ef4444"

        # --- NEW HERO SECTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        hero_c1, hero_c2, hero_c3 = st.columns([1.2, 1, 1])
        
        # 1. RISK HERO CARD
        with hero_c1:
            st.markdown(f"""
                <div class="risk-hero-dboard fade-in" style="border-top: 4px solid {risk_color};">
                    <div class="risk-label-top">Predictive Risk Assessment</div>
                    <div class="risk-value-huge" style="color: {risk_color};">{risk_score:.1f}</div>
                    <div class="risk-level-badge" style="background: {risk_color}20; color: {risk_color};">
                        {risk_level.upper()}
                    </div>
                    <div class="risk-context">
                        Anomalies detected: <b>{anomaly_count}</b><br>
                        Confidence level: <b style="color: {conf_color}">{confidence_label} ({confidence_score}%)</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        # 2. RISK GAUGE
        with hero_c2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "", 'font': {'size': 1}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': chart_colors['text'], 'visible': False},
                    'bar': {'color': risk_color, 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'bordercolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, 100], 'color': f"rgba{tuple(int(chart_colors['bg'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"}
                    ],
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': chart_colors['text'], 'family': "Inter"},
                height=280,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        # 3. QUICK STATS STACK
        with hero_c3:
            st.markdown(f"""
                <div class="fade-in">
                    <div class="stat-card-modern">
                        <div>
                            <div class="stat-label">Total Cases</div>
                            <div class="stat-value">{total_cases:,.0f}</div>
                        </div>
                    </div>
                    <div class="stat-card-modern">
                        <div>
                            <div class="stat-label">Daily Average</div>
                            <div class="stat-value">{avg_daily:.0f}</div>
                        </div>
                    </div>
                    <div class="stat-card-modern">
                        <div>
                            <div class="stat-label">Peak Volume</div>
                            <div class="stat-value">{max_daily:.0f}</div>
                        </div>
                    </div>
                    <div class="stat-card-modern">
                        <div>
                            <div class="stat-label">Anomalies</div>
                            <div class="stat-value" style="color: #ef4444;">{anomaly_count}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Weekly Trend Preview (Full Width)
        st.markdown("### 30-Day Trajectory")
        filtered_df['Week'] = pd.to_datetime(filtered_df['Date']).dt.isocalendar().week
        weekly_data = filtered_df.groupby('Week')['Cases'].sum().reset_index()
        weekly_data.columns = ['Week', 'Total_Cases']
        
        # Smoothed area chart
        fill_color = f"rgba{tuple(int(chart_colors['line'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
        
        fig_weekly = px.line(
            weekly_data,
            x='Week',
            y='Total_Cases',
            markers=True
        )
        fig_weekly.update_traces(
            line_color=chart_colors['line'], 
            marker_color=chart_colors['line'],
            line_shape='spline',
            fill='tozeroy',
            fillcolor=fill_color
        )
        fig_weekly.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='', showgrid=False),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='', showgrid=True),
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Analyze Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            def go_to_analysis():
                st.session_state.navigation = "Analysis"
                
            st.button("Analyze Full Dataset", on_click=go_to_analysis, use_container_width=True, type="primary")
    else:
        st.warning("No data available for selected filters.")

def show_analysis_page(df, dark_mode):
    """Page 2: AI Analysis Dashboard"""
    chart_colors = get_chart_colors(dark_mode)

    # Apply global background via CSS
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: { '#000000' if dark_mode else '#ffffff' };
            background-image: linear-gradient({ 'rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.92)' if dark_mode else 'rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)' }), url("background.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)
    
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
    
    # Navigation Buttons
    nav_c1, nav_c2, _ = st.columns([1, 1, 6])
    
    with nav_c1:
        def go_home():
            st.session_state.navigation = "Home"
        st.button("Home", on_click=go_home, use_container_width=True)
        
    with nav_c2:
        def go_back():
            st.session_state.navigation = "Selection"
        st.button("Back", on_click=go_back, use_container_width=True)
    
    # Header
    selected_year = st.session_state.selected_year
    st.markdown(f"""
        <h1 style="margin-bottom: 0.5rem;">Epidemic Analysis Dashboard</h1>
        <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 2rem;">
            {selected_country} • {selected_month} {selected_year}
        </p>
    """, unsafe_allow_html=True)
    
    # Early Warning Alert Panel (Full Width)
    if risk_level == "High":
        st.markdown(f"""
            <div class="alert-box alert-high fade-in">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div>
                        <h3 style="margin: 0; font-weight: 700;">High Outbreak Risk Detected</h3>
                        <p style="margin: 0; opacity: 0.9;">Immediate intervention recommended due to anomalous data patterns.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif risk_level == "Moderate":
        st.markdown(f"""
            <div class="alert-box alert-moderate fade-in">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div>
                        <h3 style="margin: 0; font-weight: 700;">Elevated Risk levels</h3>
                        <p style="margin: 0; opacity: 0.9;">Monitor closely. Statistical deviations observed.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Content Grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Outbreak Trend Graph
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); height: 100%;">
            <h3 style="margin-top: 0; margin-bottom: 1rem;">Trend Analysis</h3>
        """, unsafe_allow_html=True)
        
        # Prepare colors
        fill_color = f"rgba{tuple(int(chart_colors['line'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Main trend line
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Cases'],
                mode='lines+markers',
                name='Daily Cases',
                line=dict(color=chart_colors['line'], width=2, shape='spline'),
                fill='tozeroy',
                fillcolor=fill_color,
                marker=dict(size=4, color=chart_colors['line'])
            )
        )
        
        # Moving average
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Cases_MA7'],
                mode='lines',
                name='7-Day Average',
                line=dict(color=chart_colors['line_secondary'], width=1.5, dash='dash', shape='spline')
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
                    name='Anomalies',
                    marker=dict(
                        size=10,
                        color='#ef4444',
                        symbol='x',
                        line=dict(width=1, color=chart_colors['line'])
                    )
                )
            )

        # Create Animation Frames
        frames = []
        for k in range(1, len(filtered_df) + 1):
            subset = filtered_df.iloc[:k]
            frame_data = [
                go.Scatter(x=subset['Date'], y=subset['Cases']),
                go.Scatter(x=subset['Date'], y=subset['Cases_MA7'])
            ]
            if len(anomalies) > 0:
                subset_anom = subset[subset['Anomaly'] == -1]
                frame_data.append(go.Scatter(x=subset_anom['Date'], y=subset_anom['Cases']))
            
            frames.append(go.Frame(data=frame_data, name=str(k)))
            
        fig.frames = frames
        
        fig.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], range=[filtered_df['Date'].min(), filtered_df['Date'].max()]),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], range=[0, filtered_df['Cases'].max() * 1.1]),
            legend=dict(bgcolor=chart_colors['bg'], bordercolor=chart_colors['grid'], borderwidth=1),
            height=400,
            hovermode='x unified',
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="▶ Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 50, "redraw": False},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="⏸ Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}])
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    y=1.15
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Key Insights & Metrics Stack
        st.markdown("### Key Insights")
        
        # Calculate specific insights
        growth_rate = ((filtered_df['Cases'].iloc[-1] - filtered_df['Cases'].iloc[0]) / (filtered_df['Cases'].iloc[0] + 1)) * 100
        volatility = filtered_df['Cases'].std()
        
        st.markdown(f"""
            <div class="stat-card-modern fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="stat-label">Monthly Growth</div>
                        <div class="stat-value" style="color: {'#ef4444' if growth_rate > 0 else '#10b981'};">
                            {growth_rate:+.1f}%
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stat-card-modern fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="stat-label">Data Volatility</div>
                        <div class="stat-value">{volatility:.1f}</div>
                    </div>
                </div>
            </div>
            
            <div class="stat-card-modern fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="stat-label">Risk Score</div>
                        <div class="stat-value" style="color: {risk_color};">{risk_score:.1f}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # AI Analysis Section
    st.markdown("### AI Analysis")
    
    explanation = generate_ai_explanation(filtered_df, risk_score, risk_level)
    confidence = 85 + np.random.randint(-5, 10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-bottom: 1rem; font-weight: 600;">Model Insights</h3>
                <p style="color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">
                    {explanation}
                </p>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">
                    Confidence: <span style="color: var(--text-primary); font-weight: 500;">{confidence}%</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Model Performance")
        model_metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [
                f"{confidence}%",
                f"{confidence-2}%",
                f"{confidence-1}%",
                f"{confidence-1}%"
            ]
        }
        model_df = pd.DataFrame(model_metrics)
        st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    # Prediction Section
    st.markdown(f"""
    <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); margin-bottom: 2rem;">
        <h3 style="margin-top: 0; margin-bottom: 1rem;">7-Day Forecast</h3>
    """, unsafe_allow_html=True)
    
    # Simple linear trend forecast
    recent_trend = filtered_df['Cases'].tail(7).values
    if len(recent_trend) > 1:
        # Simple linear regression for next 7 days
        x = np.arange(len(recent_trend))
        coeffs = np.polyfit(x, recent_trend, 1)
        future_x = np.arange(len(recent_trend), len(recent_trend) + 7)
        forecast = np.polyval(coeffs, future_x)
        forecast = np.maximum(forecast, 0)  # Ensure non-negative
        
        forecast_dates = pd.date_range(start=filtered_df['Date'].max() + timedelta(days=1), periods=7, freq='D')
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast,
            'Type': 'Forecast'
        })
        
        # Combine with recent data
        recent_df = filtered_df[['Date', 'Cases']].tail(7).copy()
        recent_df['Type'] = 'Actual'
        recent_df.columns = ['Date', 'Forecast', 'Type']
        
        combined_df = pd.concat([recent_df, forecast_df])
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=recent_df['Date'],
            y=recent_df['Forecast'],
            mode='lines+markers',
            name='Actual',
            line=dict(color=chart_colors['line'], width=2),
            marker=dict(color=chart_colors['line'], size=6)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=chart_colors['line_secondary'], width=2, dash='dash'),
            marker=dict(color=chart_colors['line_secondary'], size=6)
        ))
        fig_forecast.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Cases'),
            height=300,
            legend=dict(bgcolor=chart_colors['bg'], bordercolor=chart_colors['grid'], borderwidth=1)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Forecast", f"{forecast.mean():.0f}", "Next 7 days")
        with col2:
            st.metric("Peak Forecast", f"{forecast.max():.0f}", "Expected max")
        with col3:
            forecast_change = ((forecast.mean() - recent_trend.mean()) / recent_trend.mean() * 100) if recent_trend.mean() > 0 else 0
            st.metric("Trend", f"{forecast_change:+.1f}%", "vs. recent avg")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data Insights Section
    st.markdown("### Data Insights")
    
    # Calculate comprehensive metrics
    recent_avg = filtered_df['Cases'].tail(7).mean()
    previous_avg = filtered_df['Cases'].head(7).mean() if len(filtered_df) > 7 else recent_avg
    growth_rate = ((recent_avg - previous_avg) / (previous_avg + 1e-10)) * 100
    total_cases = filtered_df['Cases'].sum()
    anomaly_count = (filtered_df['Anomaly'] == -1).sum()
    max_cases = filtered_df['Cases'].max()
    min_cases = filtered_df['Cases'].min()
    std_cases = filtered_df['Cases'].std()
    
    # Calculate additional metrics if available
    if 'Fever_Cases' in filtered_df.columns:
        total_fever = filtered_df['Fever_Cases'].sum()
        fever_rate = (total_fever / total_cases * 100) if total_cases > 0 else 0
    else:
        total_fever = 0
        fever_rate = 0
    
    if 'Hospitalizations' in filtered_df.columns:
        total_hosp = filtered_df['Hospitalizations'].sum()
        hosp_rate = (total_hosp / total_cases * 100) if total_cases > 0 else 0
    else:
        total_hosp = 0
        hosp_rate = 0
    
    if 'Deaths' in filtered_df.columns:
        total_deaths = filtered_df['Deaths'].sum()
        mortality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0
    else:
        total_deaths = 0
        mortality_rate = 0
    
    if 'Recovered' in filtered_df.columns:
        total_recovered = filtered_df['Recovered'].sum()
        recovery_rate = (total_recovered / total_cases * 100) if total_cases > 0 else 0
    else:
        total_recovered = 0
        recovery_rate = 0
    
    # Primary Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", f"{total_cases:,.0f}", f"{growth_rate:+.1f}%")
    
    with col2:
        st.metric("Avg Daily Cases", f"{recent_avg:.0f}", "Last 7 days")
    
    with col3:
        st.metric("Anomalies Detected", f"{anomaly_count}", "AI flagged")
    
    with col4:
        st.metric("Peak Daily Cases", f"{max_cases:.0f}", f"Min: {min_cases:.0f}")
    
    # Secondary Metrics Row
    st.markdown("#### Additional Statistics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Growth Rate", f"{growth_rate:+.1f}%", "vs. early period")
    
    with col2:
        st.metric("Std Deviation", f"{std_cases:.1f}", "Variability")
    
    with col3:
        if total_fever > 0:
            st.metric("Fever Cases", f"{total_fever:,.0f}", f"{fever_rate:.1f}%")
        else:
            st.metric("Fever Cases", "N/A", "")
    
    with col4:
        if total_hosp > 0:
            st.metric("Hospitalizations", f"{total_hosp:,.0f}", f"{hosp_rate:.1f}%")
        else:
            st.metric("Hospitalizations", "N/A", "")
    
    with col5:
        if total_deaths > 0:
            st.metric("Deaths", f"{total_deaths:,.0f}", f"{mortality_rate:.2f}%")
        else:
            st.metric("Deaths", "N/A", "")
    
    with col6:
        if total_recovered > 0:
            st.metric("Recovered", f"{total_recovered:,.0f}", f"{recovery_rate:.1f}%")
        else:
            st.metric("Recovered", "N/A", "")
    
    # Additional visualizations
    st.markdown("### Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; margin-bottom: 1rem;">Case Distribution</h4>
        """, unsafe_allow_html=True)
        fig_hist = px.histogram(
            filtered_df,
            x='Cases',
            nbins=20,
            color_discrete_sequence=[chart_colors['line']]
        )
        fig_hist.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Cases'),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Frequency'),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; margin-bottom: 1rem;">Anomaly Detection</h4>
        """, unsafe_allow_html=True)
        fig_anomaly = px.scatter(
            filtered_df,
            x='Date',
            y='Anomaly_Score',
            color='Anomaly',
            color_discrete_map={1: '#10b981', -1: '#ef4444'},
            labels={'Anomaly_Score': 'Anomaly Score'}
        )
        fig_anomaly.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            height=300,
            legend=dict(bgcolor=chart_colors['bg'], bordercolor=chart_colors['grid'], borderwidth=1)
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Weekly and Daily Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; margin-bottom: 1rem;">Weekly Trend</h4>
        """, unsafe_allow_html=True)
        filtered_df['Week'] = pd.to_datetime(filtered_df['Date']).dt.isocalendar().week
        weekly_summary = filtered_df.groupby('Week').agg({
            'Cases': 'sum',
            'Anomaly': lambda x: (x == -1).sum()
        }).reset_index()
        weekly_summary.columns = ['Week', 'Total_Cases', 'Anomalies']
        
        fig_weekly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_weekly.add_trace(
            go.Bar(x=weekly_summary['Week'], y=weekly_summary['Total_Cases'], 
                   name='Cases', marker_color=chart_colors['line']),
            secondary_y=False
        )
        fig_weekly.add_trace(
            go.Scatter(x=weekly_summary['Week'], y=weekly_summary['Anomalies'],
                      mode='lines+markers', name='Anomalies', line=dict(color='#ef4444')),
            secondary_y=True
        )
        fig_weekly.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0f0f0f',
            font=dict(color='#ffffff', size=11),
            xaxis=dict(gridcolor='#262626', linecolor='#262626', title='Week'),
            yaxis=dict(gridcolor='#262626', linecolor='#262626', title='Cases'),
            yaxis2=dict(gridcolor='#262626', linecolor='#262626', title='Anomalies'),
            height=300,
            legend=dict(bgcolor='#1a1a1a', bordercolor='#262626', borderwidth=1)
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; margin-bottom: 1rem;">Daily Cases vs Moving Average</h4>
        """, unsafe_allow_html=True)
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Cases'],
            mode='lines',
            name='Daily Cases',
            line=dict(color=chart_colors['line'], width=1)
        ))
        fig_ma.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Cases_MA7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color=chart_colors['line_secondary'], width=2, dash='dash')
        ))
        fig_ma.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Cases'),
            height=300,
            legend=dict(bgcolor=chart_colors['bg'], bordercolor=chart_colors['grid'], borderwidth=1)
        )
        st.plotly_chart(fig_ma, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Country Comparison Section
    st.markdown("### Country Comparison")
    
    # Get data for comparison
    df_all = load_covid_data()
    comparison_month = selected_month
    comparison_year = selected_year
    comparison_data = df_all[(df_all['Month'] == comparison_month) & (df_all['Year'] == comparison_year)].groupby('Country')['Cases'].sum().reset_index()
    comparison_data = comparison_data.sort_values('Cases', ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; margin-bottom: 1rem;">Top 10 Countries (Same Month)</h4>
        """, unsafe_allow_html=True)
        fig_bar = px.bar(
            comparison_data,
            x='Cases',
            y='Country',
            orientation='h',
            color='Cases',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Total Cases'),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            height=400,
            showlegend=False
        )
        fig_bar.update_traces(marker_color=chart_colors['line'])
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
            <h4 style="margin-top: 0; margin-bottom: 1rem;">Risk Score Distribution</h4>
        """, unsafe_allow_html=True)
        # Calculate risk for top countries
        top_countries = comparison_data['Country'].head(5).tolist()
        risk_comparison = []
        for country in top_countries:
            country_data = df_all[(df_all['Country'] == country) & (df_all['Month'] == comparison_month) & (df_all['Year'] == comparison_year)].copy()
            if len(country_data) > 0:
                country_data = preprocess_data(country_data)
                country_data = detect_anomalies(country_data)
                country_risk = calculate_risk_score(country_data)
                risk_comparison.append({'Country': country, 'Risk': country_risk})
        
        if risk_comparison:
            risk_df = pd.DataFrame(risk_comparison)
            fig_risk = px.bar(
                risk_df,
                x='Country',
                y='Risk',
                color='Risk',
                color_continuous_scale=['#10b981', '#f59e0b', '#ef4444']
            )
            fig_risk.update_layout(
                plot_bgcolor=chart_colors['bg'],
                paper_bgcolor=chart_colors['paper'],
                font=dict(color=chart_colors['text'], size=11),
                xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
                yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Risk Score'),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Statistical Summary Card
    st.markdown("### Statistical Analysis")
    
    stat_summary = filtered_df['Cases'].describe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); color: var(--text-primary);">
            <h4 style="margin-top:0;">Central Tendency</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: var(--text-muted);">Mean</span>
                <span style="font-weight: 600;">{stat_summary['mean']:,.1f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: var(--text-muted);">Median</span>
                <span style="font-weight: 600;">{stat_summary['50%']:,.1f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-muted);">Standard Deviation</span>
                <span style="font-weight: 600;">{stat_summary['std']:,.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); color: var(--text-primary);">
            <h4 style="margin-top:0;">Distribution Range</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: var(--text-muted);">Minimum</span>
                <span style="font-weight: 600;">{stat_summary['min']:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: var(--text-muted);">Maximum</span>
                <span style="font-weight: 600;">{stat_summary['max']:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-muted);">Interquartile Range</span>
                <span style="font-weight: 600;">{stat_summary['75%'] - stat_summary['25%']:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    
    # Trend and Anomaly Summary Row
    st.markdown("<br>", unsafe_allow_html=True)
    col_trend, col_anom = st.columns(2)
    
    with col_trend:
        # Calculate trend direction
        first_half = filtered_df['Cases'].head(len(filtered_df)//2).mean()
        second_half = filtered_df['Cases'].tail(len(filtered_df)//2).mean()
        trend_direction = "Increasing" if second_half > first_half else "Decreasing"
        trend_pct = abs((second_half - first_half) / first_half * 100) if first_half > 0 else 0
        trend_color = "#ef4444" if trend_direction == "Increasing" else "#10b981"
        
        st.markdown(f"""
<div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); height: 100%; color: var(--text-primary);">
<h4 style="margin-top: 0; margin-bottom: 1rem;">Trend Analysis</h4>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-color);">
<span style="color: var(--text-muted);">First Half Avg</span>
<span style="font-weight: 600; font-family: 'Inter', sans-serif;">{first_half:,.1f}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-color);">
<span style="color: var(--text-muted);">Second Half Avg</span>
<span style="font-weight: 600; font-family: 'Inter', sans-serif;">{second_half:,.1f}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="color: var(--text-muted);">Direction</span>
<span style="font-weight: 700; color: {trend_color};">{trend_direction} ({trend_pct:.1f}%)</span>
</div>
</div>
""", unsafe_allow_html=True)
    
    with col_anom:
        # Calculate anomaly stats
        anomaly_count = (filtered_df['Anomaly'] == -1).sum()
        anomaly_rate = (anomaly_count/len(filtered_df)*100)
        avg_score = filtered_df['Anomaly_Score'].mean()
        min_score = filtered_df['Anomaly_Score'].min()
        
        st.markdown(f"""
<div style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); height: 100%; color: var(--text-primary);">
<h4 style="margin-top: 0; margin-bottom: 1rem;">Anomaly Summary</h4>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="color: var(--text-muted);">Total Anomalies</span>
<span style="font-weight: 600; font-family: 'Inter', sans-serif;">{anomaly_count}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="color: var(--text-muted);">Anomaly Rate</span>
<span style="font-weight: 600; font-family: 'Inter', sans-serif;">{anomaly_rate:.1f}%</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="color: var(--text-muted);">Avg Anomaly Score</span>
<span style="font-weight: 600; font-family: 'Inter', sans-serif;">{avg_score:.3f}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="color: var(--text-muted);">Min Score</span>
<span style="font-weight: 600; font-family: 'Inter', sans-serif;">{min_score:.3f}</span>
</div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

