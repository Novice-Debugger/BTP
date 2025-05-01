import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from newsapi import NewsApiClient
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set custom theme and styling
st.set_page_config(
    page_title="Multi-Algorithm Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling with better text contrast
st.markdown("""
<style>
    /* Main color palette - modern blue/teal theme */
    :root {
        --primary: #2E86C1;
        --primary-light: #AED6F1;
        --secondary: #17A589;
        --accent: #3498DB;
        --background: #F8F9F9;
        --text: #2C3E50;
        --text-on-white: #1A1A1A; /* New darker text color for white backgrounds */
        --success: #27AE60;
        --warning: #F39C12;
        --danger: #E74C3C;
    }
    
    /* Base styling */
    .main {
        background-color: rgb(14, 17, 23);
        color: var(--text);
    }
    
    h1, h2, h3 {
        color: #AED6F1;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--accent);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
        color: var(--text-on-white); /* Ensure dark text on white background */
    }
    
    .card h4, .card h3, .card h2 {
        color: var(--text-on-white); /* Darker headers within cards */
    }
    
    .card p, .card span, .card div {
        color: var(--text-on-white); /* Ensure all text elements in cards are dark */
    }
    
    .card-success {
        border-left: 4px solid var(--success);
    }
    
    .card-warning {
        border-left: 4px solid var(--warning);
    }
    
    .card-danger {
        border-left: 4px solid var(--danger);
    }
    
    /* Metric styling */
    .metric-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        color: var(--text-on-white); /* Ensure dark text */
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary); /* Keep primary color for emphasis but ensure it's visible */
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-on-white); /* Darker text for better readability */
        opacity: 0.9; /* Slightly increased opacity */
    }
    
    /* News card styling */
    .news-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.5rem;
        transition: all 0.2s;
        border-left: 3px solid var(--secondary);
        color: var(--text-on-white); /* Ensure dark text */
    }
    
    .news-card h4 {
        color: var(--text-on-white); /* Darker text for headers */
    }
    
    .news-card p {
        color: var(--text-on-white); /* Darker text for paragraphs */
    }
    
    .news-card a {
        color: var(--primary); /* Keep links colored but ensure visibility */
        font-weight: 500; /* Make links slightly bolder */
    }
    
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .header {
        padding: 1rem 0;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Input field styling */
    div[data-baseweb="input"] {
        border-radius: 4px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: var(--primary);
    }
    
    /* Disclaimer styling */
    .disclaimer {
        font-size: 0.8rem;
        color: #5D6D7E; /* Darker color for better readability */
        font-style: italic;
        text-align: center;
        padding: 0.5rem;
        background-color: rgba(0, 0, 0, 0.03);
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    /* Signal indicators */
    .signal-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
        text-align: center;
    }
    
    .signal-buy {
        background-color: rgba(39, 174, 96, 0.2);
        color: #1e8449; /* Darker green for better readability */
        border: 1px solid #27AE60;
    }
    
    .signal-sell {
        background-color: rgba(231, 76, 60, 0.2);
        color: #b83227; /* Darker red for better readability */
        border: 1px solid #E74C3C;
    }
    
    .signal-hold {
        background-color: rgba(52, 152, 219, 0.2);
        color: #1a5276; /* Darker blue for better readability */
        border: 1px solid #3498DB;
    }
    
    /* Fix for any inline styles that might use white text */
    [style*="color: white"] {
        color: var(--text-on-white) !important;
    }
    
    [style*="color: #fff"] {
        color: var(--text-on-white) !important;
    }
    
    [style*="color: #ffffff"] {
        color: var(--text-on-white) !important;
    }
    
    /* Additional fixes for dynamic content */
    div.stMarkdown p {
        color: var(--text-on-white);
    }
    
    div.card div, div.metric-container div, div.news-card div {
        color: var(--text-on-white);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown(
    """
    <div class="header">
        <h1>Multi-Algorithm Stock Predictor</h1>
        <p>Advanced stock analysis and prediction using multiple AI models</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Disclaimer
st.markdown(
    """
    <div class="disclaimer">
        Disclaimer: This application provides stock predictions based on algorithms and is intended for informational purposes only. 
        Predictions may not be accurate, and users are encouraged to conduct their own research and consider consulting with a 
        financial advisor before making any investment decisions. This is not financial advice, and I am not responsible for any 
        outcomes resulting from the use of this application.
    </div>
    """,
    unsafe_allow_html=True
)

# API setup
NEWS_API_KEY = '0de37ca8af9748898518daf699189abf'
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Cache functions
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

@st.cache_data(ttl=3600)
def get_news_headlines(symbol):
    try:
        news = newsapi.get_everything(
            q=symbol,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        return [(article['title'], article['description'], article['url']) 
                for article in news['articles']]
    except Exception as e:
        print(f"News API error: {str(e)}")
        return []

def calculate_technical_indicators_for_summary(df):
        analysis_df = df.copy()
        
        # Calculate Moving Averages
        analysis_df['MA20'] = analysis_df['Close'].rolling(window=20).mean()
        analysis_df['MA50'] = analysis_df['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = analysis_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        analysis_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Volume MA
        analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=20).mean()
        
        # Calculate Bollinger Bands
        ma20 = analysis_df['Close'].rolling(window=20).mean()
        std20 = analysis_df['Close'].rolling(window=20).std()
        analysis_df['BB_upper'] = ma20 + (std20 * 2)
        analysis_df['BB_lower'] = ma20 - (std20 * 2)
        analysis_df['BB_middle'] = ma20
        
        return analysis_df

class MultiAlgorithmStockPredictor:
    def __init__(self, symbol=None, training_years=5, weights=None):
        self.symbol = symbol
        self.training_years = training_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weights = weights if weights is not None else WEIGHT_CONFIGURATIONS["Default"]
        
    def fetch_historical_data(self):
        # Same as original EnhancedStockPredictor
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.training_years)
        
        try:
            df = yf.download(self.symbol, start=start_date, end=end_date)
            if df.empty:
                st.warning(f"Data for the last {self.training_years} years is unavailable. Fetching maximum available data instead.")
                df = yf.download(self.symbol, period="max")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return yf.download(self.symbol, period="max")

    # Technical indicators calculation methods remain the same
    def calculate_technical_indicators(self, df):
        # Original technical indicators remain the same
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        df['ATR'] = self.calculate_atr(df)
        df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Additional technical indicators
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MOM'] = df['Close'].diff(10)
        df['STOCH_K'] = self.calculate_stochastic(df)
        df['WILLR'] = self.calculate_williams_r(df)
        
        return df.dropna()
    
    @staticmethod
    def calculate_stochastic(df, period=14):
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        return k

    @staticmethod
    def calculate_williams_r(df, period=14):
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        return -100 * ((high_max - df['Close']) / (high_max - low_min))

    # Original calculation methods remain the same
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
    
    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band

    def prepare_data(self, df, seq_length=60):
        feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                          'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                          'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Prepare sequences for LSTM
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i, 0])  # 0 index represents Close price
            
        # Prepare data for other models
        X_other = scaled_data[seq_length:]
        
        return np.array(X_lstm), X_other, np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_arima(self, df):
        model = ARIMA(df['Close'], order=(5,1,0))
        return model.fit()

    def predict_with_all_models(self, prediction_days=30, sequence_length=60):
        try:
            # Fetch and prepare data
            df = self.fetch_historical_data()
            
            # Check if we have enough data
            if len(df) < sequence_length + 20:  # Need extra days for technical indicators
                st.error(f"Insufficient historical data. Need at least {sequence_length + 20} days of data.")
                return None
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Check for NaN values and handle them
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
                
            # Verify we have enough valid data after cleaning
            if len(df.dropna()) < sequence_length:
                st.error("Insufficient valid data after calculating indicators.")
                return None
                
            # Prepare features
            feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                            'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                            'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
                            
            # Verify all required features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                return None
                
            # Ensure we have valid data for all features
            df = df[feature_columns].dropna()
            if len(df) < sequence_length:
                st.error(f"Insufficient valid data points after cleaning. Need at least {sequence_length} points.")
                st.write(f"Available data points: {len(df)}")
                return None
                
            try:
                # Scale features
                scaled_data = self.scaler.fit_transform(df[feature_columns])
            except ValueError as e:
                st.error(f"Scaling error: {str(e)}")
                st.write("This usually happens with newly listed stocks or stocks with insufficient trading history.")
                return None
                
            # Prepare sequences for LSTM
            X_lstm, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X_lstm.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])  # 0 index represents Close price
                
            # Verify we have enough sequences
            if len(X_lstm) == 0 or len(y) == 0:
                st.error("Could not create valid sequences for prediction.")
                return None
                
            # Prepare data for other models
            X_other = scaled_data[sequence_length:]
            
            # Convert to numpy arrays
            X_lstm = np.array(X_lstm)
            X_other = np.array(X_other)
            y = np.array(y)
            
            # Split data
            split_idx = int(len(y) * 0.8)
            X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
            X_other_train, X_other_test = X_other[:split_idx], X_other[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            predictions = {}
            
            # Train and predict with LSTM
            lstm_model = self.build_lstm_model((sequence_length, X_lstm.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lstm_model.fit(X_lstm_train, y_train, epochs=50, batch_size=32,
                          validation_data=(X_lstm_test, y_test),
                          callbacks=[early_stopping], verbose=0)
            lstm_pred = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]
            predictions['LSTM'] = lstm_pred

            # Train and predict with SVR
            svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr_model.fit(X_other_train, y_train)
            svr_pred = svr_model.predict(X_other_test[-1:])
            predictions['SVR'] = svr_pred[0]

            # Train and predict with Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_other_train, y_train)
            rf_pred = rf_model.predict(X_other_test[-1:])
            predictions['Random Forest'] = rf_pred[0]

            # Train and predict with XGBoost
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
            xgb_model.fit(X_other_train, y_train)
            xgb_pred = xgb_model.predict(X_other_test[-1:])
            predictions['XGBoost'] = xgb_pred[0]

            # Train and predict with KNN
            knn_model = KNeighborsRegressor(n_neighbors=5)
            knn_model.fit(X_other_train, y_train)
            knn_pred = knn_model.predict(X_other_test[-1:])
            predictions['KNN'] = knn_pred[0]

            # Train and predict with GBM
            gbm_model = GradientBoostingRegressor(random_state=42)
            gbm_model.fit(X_other_train, y_train)
            gbm_pred = gbm_model.predict(X_other_test[-1:])
            predictions['GBM'] = gbm_pred[0]

            # Train and predict with ARIMA
            try:
                close_prices = df['Close'].values
                arima_model = ARIMA(close_prices, order=(5,1,0))
                arima_fit = arima_model.fit()
                arima_pred = arima_fit.forecast(steps=1)[0]
                arima_scaled = (arima_pred - df['Close'].mean()) / df['Close'].std()
                predictions['ARIMA'] = arima_scaled
            except Exception as e:
                st.warning(f"ARIMA prediction failed: {str(e)}")

            weights = self.weights

            # Adjust weights if some models failed
            available_models = list(predictions.keys())
            total_weight = sum(weights[model] for model in available_models)
            adjusted_weights = {model: weights[model]/total_weight for model in available_models}

            ensemble_pred = sum(pred * adjusted_weights[model] 
                              for model, pred in predictions.items())
            
            # Inverse transform predictions
            dummy_array = np.zeros((1, X_other.shape[1]))
            dummy_array[0, 0] = ensemble_pred
            final_prediction = self.scaler.inverse_transform(dummy_array)[0, 0]

            # Calculate prediction range
            individual_predictions = []
            for pred in predictions.values():
                dummy = dummy_array.copy()
                dummy[0, 0] = pred
                individual_predictions.append(
                    self.scaler.inverse_transform(dummy)[0, 0]
                )
            
            std_dev = np.std(individual_predictions)
            
            return {
                'prediction': final_prediction,
                'lower_bound': final_prediction - std_dev,
                'upper_bound': final_prediction + std_dev,
                'confidence_score': 1 / (1 + std_dev / final_prediction),
                'individual_predictions': {
                    model: pred for model, pred in zip(predictions.keys(), individual_predictions)
                }
            }

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

# Streamlit redesigned interface with sidebar
with st.sidebar:
    st.markdown("### Stock Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    display_days = st.slider("Historical Data (days)", 30, 3650, 180)
    
    st.markdown("---")
    st.markdown("### Weight Configurations")
    
    # Define different weight configurations
    WEIGHT_CONFIGURATIONS = {
        "Default": {
            'LSTM': 0.3,
            'XGBoost': 0.15,
            'Random Forest': 0.15,
            'ARIMA': 0.1,
            'SVR': 0.1,
            'GBM': 0.1,
            'KNN': 0.1
        },
        "Trend-Focused": {
            'LSTM': 0.35,
            'XGBoost': 0.20,
            'Random Forest': 0.15,
            'ARIMA': 0.10,
            'SVR': 0.08,
            'GBM': 0.07,
            'KNN': 0.05
        },
        "Statistical": {
            'LSTM': 0.20,
            'XGBoost': 0.15,
            'Random Forest': 0.15,
            'ARIMA': 0.20,
            'SVR': 0.15,
            'GBM': 0.10,
            'KNN': 0.05
        },
        "Tree-Ensemble": {
            'LSTM': 0.25,
            'XGBoost': 0.25,
            'Random Forest': 0.20,
            'ARIMA': 0.10,
            'SVR': 0.08,
            'GBM': 0.07,
            'KNN': 0.05
        },
        "Balanced": {
            'LSTM': 0.25,
            'XGBoost': 0.20,
            'Random Forest': 0.15,
            'ARIMA': 0.15,
            'SVR': 0.10,
            'GBM': 0.10,
            'KNN': 0.05
        },
        "Volatility-Focused": {
            'LSTM': 0.30,
            'XGBoost': 0.25,
            'Random Forest': 0.20,
            'ARIMA': 0.05,
            'SVR': 0.10,
            'GBM': 0.07,
            'KNN': 0.03
        }
    }

    WEIGHT_DESCRIPTIONS = {
        "Default": "Original configuration with balanced weights",
        "Trend-Focused": "Best for growth stocks, tech stocks, clear trend patterns",
        "Statistical": "Best for blue chip stocks, utilities, stable dividend stocks",
        "Tree-Ensemble": "Best for stocks with complex relationships to market factors",
        "Balanced": "Best for general purpose, unknown stock characteristics",
        "Volatility-Focused": "Best for small cap stocks, emerging market stocks, crypto-related stocks"
    }
    
    selected_weight = "Default"  # Commented out but using default
    st.info(WEIGHT_DESCRIPTIONS['Default'])
    
    st.markdown("---")
    generate_button = st.button("Generate Predictions", type="primary")

try:
    # Fetch data
    with st.spinner("Fetching stock data..."):
        df = fetch_stock_data(symbol, display_days)
    
    # Main content area
    if not df.empty:
        # Stock summary and overview in a card
        st.markdown("""
        <div class="card">
            <h2 style="color: var(--text-on-white);">Stock Overview: {}</h2>
            <p style="color: var(--text-on-white);">Analyzing historical data and generating predictive insights</p>
        </div>
        """.format(symbol), unsafe_allow_html=True)
        
        # Price history in a modern visualization
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        ))
        
        # Add moving averages
        if len(df) >= 20:
            df['MA20'] = df['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['MA20'], 
                mode='lines', 
                name='20-day MA',
                line=dict(color='#2E86C1', width=1)
            ))
        if len(df) >= 50:
            df['MA50'] = df['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['MA50'], 
                mode='lines', 
                name='50-day MA',
                line=dict(color='#17A589', width=1.5)
            ))
            
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price History',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        
        # Optimize for mobile
        fig.update_layout(hovermode='x unified')
        
        # Show the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Two-column layout for main content
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Prediction section
            if generate_button:
                with st.spinner("Training multiple models and generating predictions..."):
                    predictor = MultiAlgorithmStockPredictor(
                        symbol, 
                        weights=WEIGHT_CONFIGURATIONS['Default']
                    )
                    results = predictor.predict_with_all_models()
                    
                    if results is not None:
                        # Get current price
                        last_price = float(df['Close'].iloc[-1])
                        
                        # Calculate price change
                        price_change = ((results['prediction'] - last_price) / last_price) * 100
                        
                        # Trading signal with visual indicator
                        signal_type = "hold"
                        if abs(price_change) > 10:
                            signal_type = "buy" if price_change > 0 else "sell"
                        elif abs(price_change) > 3 and results['confidence_score'] > 0.8:
                            signal_type = "buy" if price_change > 0 else "sell"
                        elif abs(price_change) > 2 and results['confidence_score'] > 0.6:
                            signal_type = "buy" if price_change > 0 else "sell"
                        
                        # Display prediction summary in an attractive card
                        st.markdown(f"""
                        <div class="card card-{'success' if price_change > 0 else 'danger' if price_change < 0 else 'warning'}">
                            <h3 style="color: var(--text-on-white);">Prediction Summary</h3>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                                <div>
                                    <p style="color: var(--text-on-white);"><strong>Current Price:</strong> ${last_price:.2f}</p>
                                    <p style="color: var(--text-on-white);"><strong>Predicted Price:</strong> ${results['prediction']:.2f}</p>
                                </div>
                                <div>
                                    <p style="color: var(--text-on-white);"><strong>Potential Change:</strong> {price_change:+.2f}%</p>
                                    <p style="color: var(--text-on-white);"><strong>Confidence Score:</strong> {results['confidence_score']:.1%}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Trading signal box
                        signal_text = ""
                        signal_class = ""
                        
                        if abs(price_change) > 10:  # For very large changes
                            if price_change > 0:
                                signal_text = f"💹 Strong BUY Signal (+{price_change:.1f}%)"
                                signal_class = "signal-buy"
                            else:
                                signal_text = f"📉 Strong SELL Signal ({price_change:.1f}%)"
                                signal_class = "signal-sell"
                        elif abs(price_change) > 3 and results['confidence_score'] > 0.8:
                            if price_change > 0:
                                signal_text = f"💹 BUY Signal (+{price_change:.1f}%)"
                                signal_class = "signal-buy"
                            else:
                                signal_text = f"📉 SELL Signal ({price_change:.1f}%)"
                                signal_class = "signal-sell"
                        elif abs(price_change) > 2 and results['confidence_score'] > 0.6:
                            if price_change > 0:
                                signal_text = f"📈 Moderate BUY Signal (+{price_change:.1f}%)"
                                signal_class = "signal-buy"
                            else:
                                signal_text = f"📉 Moderate SELL Signal ({price_change:.1f}%)"
                                signal_class = "signal-sell"
                        else:
                            if abs(price_change) < 1:
                                signal_text = f"⚖ HOLD Signal ({price_change:.1f}%)"
                                signal_class = "signal-hold"
                            else:
                                if price_change > 0:
                                    signal_text = f"📈 Weak BUY Signal (+{price_change:.1f}%)"
                                    signal_class = "signal-buy"
                                else:
                                    signal_text = f"📉 Weak SELL Signal ({price_change:.1f}%)"
                                    signal_class = "signal-sell"
                        
                        st.markdown(f'<div class="signal-box {signal_class}">{signal_text}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Create visualization with a single combined bar instead of individual model bars
                        predictions = list(results['individual_predictions'].values())
                        models = list(results['individual_predictions'].keys())

                        # Enhanced visualization with Plotly
                        fig = go.Figure()

                        # Calculate average prediction for the single bar
                        avg_prediction = results['prediction']  # This is already the ensemble prediction

                        # Add a single horizontal bar for the ensemble prediction
                        fig.add_trace(go.Bar(
                            y=['Ensemble Prediction'],
                            x=[avg_prediction],
                            orientation='h',
                            marker=dict(
                                color='#3498DB',
                                line=dict(color='rgba(0, 0, 0, 0.1)', width=1)
                            ),
                            name='Ensemble Prediction'
                        ))

                        # Add lines for current price and ensemble prediction
                        fig.add_vline(x=last_price, line_width=2, line_dash="dash", line_color="#E74C3C",
                                     annotation_text="Current Price", annotation_position="top right")

                        # Update layout
                        fig.update_layout(
                            title="Ensemble Prediction vs Current Price",
                            xaxis_title="Price ($)",
                            height=400,
                            template="plotly_white",
                            margin=dict(l=0, r=0, t=50, b=0)
                        )

                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model consensus analysis
                        st.subheader("Model Consensus Analysis")
                        
                        # Calculate consensus metrics
                        buy_signals = sum(1 for pred in predictions if pred > last_price)
                        sell_signals = sum(1 for pred in predictions if pred < last_price)
                        total_models = len(predictions)
                        consensus_strength = abs(buy_signals - sell_signals) / total_models
                        
                        # Create 3-column metrics display
                        consensus_cols = st.columns(3)
                        with consensus_cols[0]:
                            st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value" style="color: var(--primary);">{}/{}</div>
                                <div class="metric-label" style="color: var(--text-on-white);">Buy Signals</div>
                            </div>
                            """.format(buy_signals, total_models), unsafe_allow_html=True)
                        
                        with consensus_cols[1]:
                            st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value" style="color: var(--primary);">{}/{}</div>
                                <div class="metric-label" style="color: var(--text-on-white);">Sell Signals</div>
                            </div>
                            """.format(sell_signals, total_models), unsafe_allow_html=True)
                        
                        with consensus_cols[2]:
                            st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value" style="color: var(--primary);">{:.1%}</div>
                                <div class="metric-label" style="color: var(--text-on-white);">Consensus Strength</div>
                            </div>
                            """.format(consensus_strength), unsafe_allow_html=True)
                        
                        # Risk assessment
                        st.subheader("Risk Assessment")
                        
                        # Calculate risk metrics
                        prediction_std = np.std(predictions)
                        prediction_range = results['upper_bound'] - results['lower_bound']
                        risk_level = "Low" if prediction_std < last_price * 0.02 else \
                                    "Medium" if prediction_std < last_price * 0.05 else "High"
                        
                        # Create 2-column metrics display for risk
                        risk_cols = st.columns(2)
                        with risk_cols[0]:
                            st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value" style="color: var(--primary);">${:.2f}</div>
                                <div class="metric-label" style="color: var(--text-on-white);">Prediction Volatility</div>
                            </div>
                            """.format(prediction_std), unsafe_allow_html=True)
                        
                        with risk_cols[1]:
                            st.markdown("""
                            <div class="metric-container">
                                <div class="metric-value" style="color: var(--primary);">{}</div>
                                <div class="metric-label" style="color: var(--text-on-white);">Risk Level</div>
                            </div>
                            """.format(risk_level), unsafe_allow_html=True)
                    else:
                        st.error("Unable to generate predictions. Please check the stock symbol and try again.")
        
        with col2:
            # News and Technical Analysis in tabs
            tabs = st.tabs(["📰 Latest News", "📊 Technical Analysis"])
            
            with tabs[0]:
                st.subheader("Market Sentiment")
                
                news_headlines = get_news_headlines(symbol)
                
                if news_headlines:
                    for title, description, url in news_headlines:
                        st.markdown(f"""
                        <div class="news-card">
                            <h4 style="color: var(--text-on-white);">{title}</h4>
                            <p style="color: var(--text-on-white);">{description[:150]}{'...' if len(description) > 150 else ''}</p>
                            <a href="{url}" target="_blank" style="color: var(--primary);">Read full article →</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent news available for this stock.")
            
            with tabs[1]:
                # Technical Analysis Summary
                st.subheader("Technical Indicators")
                
                try:
                    # Check if dataframe exists and has data
                    if 'df' in locals() and isinstance(df, pd.DataFrame) and len(df) > 0:
                        # Calculate technical indicators from historical data
                        analysis_df = calculate_technical_indicators_for_summary(df)
                        
                        if len(analysis_df) >= 2:
                            latest = analysis_df.iloc[-1]
                            prev = analysis_df.iloc[-2]
                            
                            # Historical Data Analysis
                            st.markdown('<div class="card"><h4 style="color: var(--text-on-white);">Historical Data Analysis</h4>', unsafe_allow_html=True)
                            
                            # Calculate indicator values first to avoid Series truth value ambiguity
                            ma_bullish = float(latest['MA20']) > float(latest['MA50'])
                            rsi_value = float(latest['RSI'])
                            volume_high = float(latest['Volume']) > float(latest['Volume_MA'])
                            close_price = float(latest['Close'])
                            bb_upper = float(latest['BB_upper'])
                            bb_lower = float(latest['BB_lower'])
                            
                            # Historical indicators
                            historical_indicators = {
                                "Moving Averages": {
                                    "value": "Bullish" if ma_bullish else "Bearish",
                                    "delta": f"{((float(latest['MA20']) - float(latest['MA50']))/float(latest['MA50']) * 100):.1f}% spread",
                                    "description": "Based on 20 & 50-day moving averages"
                                },
                                "RSI (14)": {
                                    "value": "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral",
                                    "delta": f"{rsi_value:.1f}",
                                    "description": "Current RSI value"
                                },
                                "Volume Trend": {
                                    "value": "Above Average" if volume_high else "Below Average",
                                    "delta": f"{((float(latest['Volume']) - float(latest['Volume_MA']))/float(latest['Volume_MA']) * 100):.1f}%",
                                    "description": "Compared to 20-day average"
                                },
                                "Bollinger Bands": {
                                    "value": "Upper Band" if close_price > bb_upper else 
                                            "Lower Band" if close_price < bb_lower else "Middle Band",
                                    "delta": f"{((close_price - bb_lower)/(bb_upper - bb_lower) * 100):.1f}%",
                                    "description": "Position within bands"
                                }
                            }
                            
                            # Display historical indicators in a more modern way
                            for indicator, data in historical_indicators.items():
                                # Set color based on indicator
                                color_class = ""
                                if "Bullish" in data["value"] or "Above Average" in data["value"] or "Oversold" in data["value"]:
                                    color_class = "success"
                                elif "Bearish" in data["value"] or "Below Average" in data["value"] or "Overbought" in data["value"]:
                                    color_class = "danger"
                                
                                st.markdown(f"""
                                <div class="card card-{color_class}" style="margin-bottom: 0.5rem; padding: 0.75rem;">
                                    <div style="display: flex; justify-content: space-between;">
                                        <div>
                                            <strong style="color: var(--text-on-white);">{indicator}</strong><br>
                                            <small style="color: var(--text-on-white);">{data['description']}</small>
                                        </div>
                                        <div style="text-align: right;">
                                            <span style="font-size: 1.1rem; color: var(--text-on-white);">{data['value']}</span><br>
                                            <small style="color: var(--text-on-white);">{data['delta']}</small>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # If prediction results exist, show them in technical analysis too
                            if 'results' in locals() and results is not None:
                                st.markdown('<div class="card"><h4 style="color: var(--text-on-white);">Combined Signal Analysis</h4>', unsafe_allow_html=True)
                                
                                # Calculate prediction metrics
                                current_price = float(df['Close'].iloc[-1])
                                pred_price = float(results['prediction'])
                                price_change_pct = ((pred_price - current_price) / current_price) * 100
                                
                                # Get trading signal strength based on price_change
                                def get_trading_signal_strength(price_change, confidence_score):
                                    if abs(price_change) > 10:
                                        return "strong_buy" if price_change > 0 else "strong_sell"
                                    elif abs(price_change) > 3 and confidence_score > 0.8:
                                        return "buy" if price_change > 0 else "sell"
                                    elif abs(price_change) > 2 and confidence_score > 0.6:
                                        return "moderate_buy" if price_change > 0 else "moderate_sell"
                                    elif abs(price_change) < 1:
                                        return "hold"
                                    else:
                                        return "weak_buy" if price_change > 0 else "weak_sell"
                                
                                # Get signals from different sources
                                technical_bullish = ma_bullish
                                trading_signal = get_trading_signal_strength(price_change_pct, results['confidence_score'])
                                model_confidence = results['confidence_score'] > 0.6
                                
                                # Determine overall signal and display with appropriate styling
                                signal_style = "success"
                                signal_text = ""
                                
                                if technical_bullish and trading_signal in ['strong_buy', 'buy']:
                                    signal_text = "🚀 Very Strong Buy Signal: Technical analysis is bullish and models show strong upward momentum"
                                    signal_style = "success"
                                elif technical_bullish and trading_signal in ['moderate_buy', 'weak_buy']:
                                    signal_text = "💹 Strong Buy Signal: Technical analysis is bullish with moderate model support"
                                    signal_style = "success"
                                elif not technical_bullish and trading_signal in ['strong_buy', 'buy']:
                                    signal_text = "📈 Cautious Buy Signal: Models show strong upward potential but technical indicators suggest caution"
                                    signal_style = "warning"
                                elif technical_bullish and trading_signal in ['hold']:
                                    signal_text = "⚖ Hold with Bullish Bias: Technical analysis is positive but models suggest consolidation"
                                    signal_style = "primary"
                                elif not technical_bullish and trading_signal in ['hold']:
                                    signal_text = "⚖ Hold with Bearish Bias: Technical analysis is negative and models suggest consolidation"
                                    signal_style = "primary"
                                elif technical_bullish and trading_signal in ['weak_sell', 'moderate_sell']:
                                    signal_text = "🤔 Mixed Signal: Technical analysis is bullish but models show weakness"
                                    signal_style = "warning"
                                elif not technical_bullish and trading_signal in ['weak_sell', 'moderate_sell']:
                                    signal_text = "📉 Strong Sell Signal: Both technical analysis and models show weakness"
                                    signal_style = "danger"
                                elif not technical_bullish and trading_signal in ['strong_sell', 'sell']:
                                    signal_text = "🔻 Very Strong Sell Signal: Technical analysis is bearish and models show strong downward momentum"
                                    signal_style = "danger"
                                else:
                                    signal_text = "🔄 Mixed Signals: Conflicting indicators suggest caution"
                                    signal_style = "warning"
                                
                                # Display the signal in a styled box
                                st.markdown(f"""
                                <div class="card card-{signal_style}" style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
                                    <h4 style="color: var(--text-on-white);">{signal_text}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display confidence metrics
                                confidence_text = "High" if model_confidence else "Low"
                                st.info(f"Model Prediction Confidence: {confidence_text}")
                                
                                # Additional context based on signals
                                if model_confidence:
                                    if technical_bullish:
                                        st.markdown("""
                                        <div style="background-color: rgba(39, 174, 96, 0.1); padding: 0.75rem; border-radius: 4px; border-left: 3px solid #27AE60;">
                                            <p style="margin-bottom: 0; color: var(--text-on-white);">💡 Technical indicators support the model predictions, suggesting higher reliability</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown("""
                                        <div style="background-color: rgba(243, 156, 18, 0.1); padding: 0.75rem; border-radius: 4px; border-left: 3px solid #F39C12;">
                                            <p style="margin-bottom: 0; color: var(--text-on-white);">💡 Technical indicators contrast with model predictions, suggesting careful monitoring</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div style="background-color: rgba(52, 152, 219, 0.1); padding: 0.75rem; border-radius: 4px; border-left: 3px solid #3498DB;">
                                        <p style="margin-bottom: 0; color: var(--text-on-white);">💡 Lower model confidence suggests waiting for clearer signals before making decisions</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Insufficient data points for technical analysis. Please ensure you have at least 50 days of historical data.")
                    else:
                        st.warning("No data available for technical analysis. Please enter a valid stock symbol.")
                                
                except Exception as e:
                    st.error(f"Error in Technical Analysis: {str(e)}")
    else:
        st.error(f"Error fetching data for {symbol}. Please check the symbol and try again.")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Please check your input and try again.")