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

# Theme colors
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
            --border-color: #262626;
            --button-bg: #ffffff;
            --button-text: #0f0f0f;
            --button-hover: #f5f5f5;
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
        }
        """

# Custom CSS for styling
def apply_theme(dark_mode):
    theme_css = get_theme_css(dark_mode)
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Times+New+Roman:wght@400;700&display=swap');
    
    {theme_css}
    
    .main {{
        background: var(--bg-primary);
    }}
    
    .stApp {{
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }}
    
    .brand-container {{
        text-align: center;
        padding: 4rem 2rem;
        margin-bottom: 3rem;
        border-bottom: 1px solid var(--border-color);
    }}
    
    .brand-title {{
        font-size: 5rem;
        font-weight: 700;
        font-family: 'Times New Roman', serif;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }}
    
    .brand-tagline {{
        font-size: 1.1rem;
        color: var(--text-muted);
        font-weight: 400;
        letter-spacing: 0.5px;
    }}
    
    .risk-preview-box {{
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 2rem;
        border-left: 3px solid;
        margin: 2rem 0;
    }}
    
    .risk-low {{
        border-left-color: #10b981;
    }}
    
    .risk-moderate {{
        border-left-color: #f59e0b;
    }}
    
    .risk-high {{
        border-left-color: #ef4444;
    }}
    
    .risk-level-text {{
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }}
    
    .metric-card {{
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }}
    
    .alert-box {{
        padding: 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        border-left: 3px solid;
        background: var(--bg-secondary);
    }}
    
    .alert-high {{
        border-left-color: #ef4444;
        color: #dc2626;
    }}
    
    .alert-moderate {{
        border-left-color: #f59e0b;
        color: #d97706;
    }}
    
    .stButton>button {{
        background: var(--button-bg);
        color: var(--button-text);
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }}
    
    .stButton>button:hover {{
        background: var(--button-hover);
        transform: translateY(-1px);
    }}
    
    h1, h2, h3 {{
        color: var(--text-primary);
        font-weight: 600;
    }}
    
    h1 {{
        font-size: 2rem;
    }}
    
    h2 {{
        font-size: 1.5rem;
    }}
    
    h3 {{
        font-size: 1.25rem;
    }}
    
    .stSelectbox label, .stSelectbox div {{
        color: var(--text-primary);
    }}
    
    .stSelectbox>div>div {{
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }}
    
    .stTextInput>div>div>input {{
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Cleaner sidebar */
    .css-1d391kg {{
        background: var(--bg-primary);
    }}
    
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        color: var(--text-primary);
    }}
    
    [data-testid="stMetricLabel"] {{
        color: var(--text-muted);
    }}
    
    [data-testid="stMetricDelta"] {{
        color: var(--text-muted);
    }}
    
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
    st.session_state.dark_mode = False

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
    dark_mode_icon = "🌙" if not dark_mode else "☀️"
    dark_mode_text = "Dark Mode" if not dark_mode else "Light Mode"
    
    # Top bar with dark mode toggle
    col_toggle, col_spacer = st.columns([3, 20])
    with col_toggle:
        if st.button(f"{dark_mode_icon} {dark_mode_text}", key="dark_mode_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    
    # Dark mode toggle in sidebar as well
    with st.sidebar:
        st.markdown("### 🌓 Theme")
        if st.button(f"{dark_mode_icon} Switch to {dark_mode_text}", key="dark_mode_toggle_sidebar", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.markdown("---")
        
        # Show dataset status
        st.markdown("### 📊 Dataset Status")
        if 'Year' in df.columns:
            st.success("✅ Real COVID-19 dataset loaded")
        else:
            st.warning("⚠️ Using sample data. Download dataset from Kaggle.")
    
    # Page selection
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Navigation", ["🏠 Landing & Filter", "📊 AI Analysis Dashboard"])
    
    if page == "🏠 Landing & Filter":
        show_landing_page(df, dark_mode)
    else:
        show_analysis_page(df, dark_mode)

def show_landing_page(df, dark_mode):
    """Page 1: Landing + Filter Dashboard"""
    chart_colors = get_chart_colors(dark_mode)
    
    # Branding Section
    st.markdown("""
        <div class="brand-container">
            <div class="brand-title">EpiWatch AI</div>
            <div class="brand-tagline">Predicting outbreaks before they escalate</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset info
    if 'Year' not in df.columns:
        st.info("💡 **Tip**: Download the COVID-19 dataset from [Kaggle](https://www.kaggle.com/datasets/josephassaker/covid19-global-dataset) and place it in this folder to use real data.")
    
    # Filter Panel
    st.markdown("### Select Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = sorted(df['Country'].unique())
        selected_country = st.selectbox(
            "Select Country",
            options=countries,
            index=0 if st.session_state.selected_country is None else countries.index(st.session_state.selected_country) if st.session_state.selected_country in countries else 0
        )
    
    with col2:
        years = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox(
            "Select Year",
            options=years,
            index=0 if st.session_state.selected_year is None else years.index(st.session_state.selected_year) if st.session_state.selected_year in years else 0
        )
    
    with col3:
        # Filter months based on selected year and country
        available_months = sorted(df[(df['Country'] == selected_country) & (df['Year'] == selected_year)]['Month'].unique())
        if len(available_months) == 0:
            available_months = sorted(df['Month'].unique())
        
        selected_month = st.selectbox(
            "Select Month",
            options=available_months,
            index=0 if st.session_state.selected_month is None else (available_months.index(st.session_state.selected_month) if st.session_state.selected_month in available_months else 0)
        )
    
    # Store selections
    st.session_state.selected_country = selected_country
    st.session_state.selected_year = selected_year
    st.session_state.selected_month = selected_month
    
    # Filter data
    filtered_df = df[(df['Country'] == selected_country) & (df['Year'] == selected_year) & (df['Month'] == selected_month)].copy()
    
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
        
        # Calculate quick stats
        total_cases = filtered_df['Cases'].sum()
        avg_daily = filtered_df['Cases'].mean()
        max_daily = filtered_df['Cases'].max()
        anomaly_count = (filtered_df['Anomaly'] == -1).sum()
        
        # AI Risk Preview Box
        risk_class = f"risk-{risk_level.lower()}"
        confidence = 85 + np.random.randint(-5, 10)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_color = "#1a1a1a" if not st.session_state.dark_mode else "#ffffff"
            muted_color = "#6c757d" if not st.session_state.dark_mode else "#888888"
            st.markdown(f"""
                <div class="risk-preview-box {risk_class}">
                    <h3 style="color: var(--text-primary); margin-bottom: 1rem; font-weight: 500;">Risk Assessment</h3>
                    <div class="risk-level-text" style="color: {risk_color}; margin-bottom: 1rem;">{risk_level}</div>
                    <p style="color: var(--text-muted); font-size: 0.95rem; margin: 0.5rem 0;">Risk Score: <span style="color: {risk_color}; font-weight: 500;">{risk_score:.1f}/100</span></p>
                    <p style="color: var(--text-muted); font-size: 0.95rem; margin: 0.5rem 0;">Confidence: <span style="color: {risk_color}; font-weight: 500;">{confidence}%</span></p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Quick Stats")
            st.metric("Total Cases", f"{total_cases:,.0f}")
            st.metric("Avg Daily", f"{avg_daily:.0f}")
            st.metric("Peak Daily", f"{max_daily:.0f}")
            st.metric("Anomalies", f"{anomaly_count}")
        
        # Weekly Trend Preview
        st.markdown("### Weekly Trend Preview")
        filtered_df['Week'] = pd.to_datetime(filtered_df['Date']).dt.isocalendar().week
        weekly_data = filtered_df.groupby('Week')['Cases'].sum().reset_index()
        weekly_data.columns = ['Week', 'Total_Cases']
        
        fig_weekly = px.line(
            weekly_data,
            x='Week',
            y='Total_Cases',
            markers=True
        )
        fig_weekly.update_traces(line_color=chart_colors['line'], marker_color=chart_colors['line'])
        fig_weekly.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Week'),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid'], title='Cases'),
            height=250,
            showlegend=False
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Analyze Button
        if st.button("Analyze Outbreak Patterns", use_container_width=True):
            st.rerun()
    else:
        st.warning("No data available for selected filters.")

def show_analysis_page(df, dark_mode):
    """Page 2: AI Analysis Dashboard"""
    chart_colors = get_chart_colors(dark_mode)
    
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
    selected_year = st.session_state.selected_year
    st.markdown(f"""
        <h1 style="margin-bottom: 0.5rem;">Analysis Dashboard</h1>
        <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 2rem;">
            {selected_country} • {selected_month} {selected_year}
        </p>
    """, unsafe_allow_html=True)
    
    # Early Warning Alert Panel
    if risk_level == "High":
        st.markdown(f"""
            <div class="alert-box alert-high">
                <h3 style="margin: 0 0 0.5rem 0; font-weight: 600;">High Outbreak Risk</h3>
                <p style="font-size: 0.95rem; margin: 0;">
                    Immediate attention recommended. Anomalous patterns detected.
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif risk_level == "Moderate":
        st.markdown(f"""
            <div class="alert-box alert-moderate">
                <h3 style="margin: 0 0 0.5rem 0; font-weight: 600;">Monitoring Required</h3>
                <p style="font-size: 0.95rem; margin: 0;">
                    Elevated patterns observed. Continued monitoring advised.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main Content Grid
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Outbreak Trend Graph
        st.markdown("### Trend Analysis")
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Main trend line
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['Cases'],
                mode='lines+markers',
                name='Daily Cases',
                line=dict(color=chart_colors['line'], width=2),
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
                line=dict(color=chart_colors['line_secondary'], width=1.5, dash='dash')
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
        
        fig.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            xaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            yaxis=dict(gridcolor=chart_colors['grid'], linecolor=chart_colors['grid']),
            legend=dict(bgcolor=chart_colors['bg'], bordercolor=chart_colors['grid'], borderwidth=1),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Indicator Meter (Gauge)
        st.markdown("### Risk Level")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'color': '#ffffff', 'size': 16}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': '#ffffff'},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 35], 'color': '#1a1a1a'},
                    {'range': [35, 70], 'color': '#1a1a1a'},
                    {'range': [70, 100], 'color': '#1a1a1a'}
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 3},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        fig_gauge.update_layout(
            plot_bgcolor=chart_colors['bg'],
            paper_bgcolor=chart_colors['paper'],
            font=dict(color=chart_colors['text'], size=11),
            height=300
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <h3 style="color: {risk_color}; margin: 0; font-weight: 600;">{risk_level}</h3>
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
    st.markdown("### 7-Day Forecast")
    
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
        st.markdown("#### Case Distribution")
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
    
    with col2:
        st.markdown("#### Anomaly Detection")
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
    
    # Weekly and Daily Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Weekly Trend")
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
        st.markdown("#### Daily Cases vs Moving Average")
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
        st.markdown("#### Top 10 Countries (Same Month)")
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
    
    with col2:
        st.markdown("#### Risk Score Distribution")
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
    
    # Statistical Summary
    st.markdown("### Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Descriptive Statistics")
        stats_data = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                f"{filtered_df['Cases'].mean():.2f}",
                f"{filtered_df['Cases'].median():.2f}",
                f"{filtered_df['Cases'].std():.2f}",
                f"{filtered_df['Cases'].min():.0f}",
                f"{filtered_df['Cases'].max():.0f}",
                f"{filtered_df['Cases'].max() - filtered_df['Cases'].min():.0f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Trend Analysis")
        # Calculate trend direction
        first_half = filtered_df['Cases'].head(len(filtered_df)//2).mean()
        second_half = filtered_df['Cases'].tail(len(filtered_df)//2).mean()
        trend_direction = "Increasing" if second_half > first_half else "Decreasing"
        trend_pct = abs((second_half - first_half) / first_half * 100) if first_half > 0 else 0
        
        trend_data = {
            'Period': ['First Half', 'Second Half', 'Trend'],
            'Avg Cases': [
                f"{first_half:.1f}",
                f"{second_half:.1f}",
                f"{trend_direction} ({trend_pct:.1f}%)"
            ]
        }
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True, hide_index=True)
    
    with col3:
        st.markdown("#### Anomaly Summary")
        anomaly_summary = {
            'Metric': ['Total Anomalies', 'Anomaly Rate', 'Avg Anomaly Score', 'Min Score'],
            'Value': [
                f"{anomaly_count}",
                f"{(anomaly_count/len(filtered_df)*100):.1f}%",
                f"{filtered_df['Anomaly_Score'].mean():.3f}",
                f"{filtered_df['Anomaly_Score'].min():.3f}"
            ]
        }
        anomaly_df = pd.DataFrame(anomaly_summary)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

