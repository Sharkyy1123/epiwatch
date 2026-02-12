# epiwatch

# 🦠 EpiWatch AI - Outbreak Prediction Platform

An AI-driven global outbreak intelligence platform that analyzes country-level epidemiological data, detects abnormal patterns using unsupervised machine learning, and provides real-time risk alerts through an interactive web dashboard.

## 🚀 Features

### Page 1 - Landing & Filter Dashboard
- **Project Branding**: Modern health-tech themed UI with animated branding
- **Smart Search Panel**: Country and Month selection with auto-suggestions
- **AI Risk Preview**: Real-time risk level indicator with color-coded alerts
- **Instant Analysis**: Dynamic risk calculation before detailed analysis

### Page 2 - AI Analysis Dashboard
- **📈 Outbreak Trend Graph**: Interactive Plotly graph with daily cases, moving average, and AI anomaly markers
- **🧠 AI Analysis Section**: Intelligent explanations of detected patterns
- **🎯 Risk Indicator Meter**: Gauge chart (speedometer-style) showing risk level
- **🚨 Early Warning Alert Panel**: Color-coded alerts for high/moderate risk
- **📊 Data Insights**: Comprehensive metrics and visualizations

## 🧠 AI Technology

- **Time-Series Preprocessing**: Advanced data preparation
- **Isolation Forest**: Unsupervised anomaly detection algorithm
- **Composite Risk Score**: Multi-factor risk calculation
- **Threshold-Based Alerts**: Intelligent warning system

## 🛠️ Tech Stack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning (Isolation Forest)
- **NumPy**: Numerical computations

## 📦 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd epiwatch-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 🎨 Design Theme

- **Background**: Dark navy gradient (#0a0e27 to #1a1f3a)
- **Accent Colors**:
  - 🟢 Low Risk: Neon Green (#00ff96)
  - 🟡 Moderate Risk: Yellow (#ffd700)
  - 🔴 High Risk: Red (#ff4444)
- **Font**: Inter (Google Fonts)
- **Animations**: Subtle glow effects and transitions

## 📊 How It Works

1. **Select Country & Month**: Choose from the dropdown menus on the landing page
2. **View Risk Preview**: Instant AI-powered risk assessment appears
3. **Click Analyze**: Navigate to detailed dashboard
4. **Explore Insights**: View graphs, metrics, and AI explanations

## 🔍 AI Model Details

The system uses **Isolation Forest** to detect anomalies in epidemiological data by:
- Analyzing case trends
- Calculating moving averages
- Detecting abnormal growth patterns
- Computing composite risk scores

## 📝 Notes

- The app uses synthetic data for demonstration
- All calculations are performed in real-time
- Risk scores are calculated using multiple factors
- Anomalies are marked with red X markers on graphs

## 🎯 Future Enhancements

- Real-time data integration
- Multi-country comparison
- Predictive forecasting
- Export reports functionality
- Historical trend analysis

---

**Built for Hackathon Excellence** 🏆
