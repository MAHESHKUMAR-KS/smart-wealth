# WealthyWise Professional - AI-Enhanced Investment Platform

An advanced mutual fund advisory platform with real-time analytics and AI-powered insights for informed investment decisions.

## Features

### Core Functionality
- **Scientific Risk Profiling** - Not random suggestions
- **Data-Driven Recommendations** - Analyzing 800+ funds
- **Advanced SIP Calculator** - With step-up & tax planning
- **Goal-Based Planning** - Retirement, education, wealth creation
- **Performance Analytics** - Quality scores & metrics
- **Portfolio Optimization** - Diversification & rebalancing

### AI-Powered Enhancements
- **Machine Learning Predictions** - Forecast fund performance using historical data
- **Sentiment Analysis** - Analyze market news for investment insights
- **AI Recommendation Engine** - Enhanced fund selection using predictive models
- **Intelligent Portfolio Optimization** - Dynamic allocation based on market conditions
- **AI Financial Assistant** - Chatbot for investment queries and guidance
- **AI-Powered Risk Assessment** - Machine learning models to analyze user behavior patterns and refine risk profiling
- **User Clustering** - Group users with similar profiles for better recommendations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MAHESHKUMAR-KS/smart-wealth.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app_advanced.py
   ```

4. (Optional) Manually train AI models with your own data:
   ```
   python train_ai_models.py
   ```

5. (Optional) Train AI models on Kaggle:
   - Use `kaggle_risk_assessment.ipynb` notebook
   - Download trained models
   - Integrate with `download_kaggle_models.py`

## AI Components

### 1. Fund Performance Predictor (`ai_predictor.py`)
Uses Random Forest Regression to predict future fund performance based on:
- Historical returns (1Y, 3Y, 5Y)
- Risk metrics (Sharpe, Sortino, Alpha, Beta)
- Fund characteristics (expense ratio, age, rating)

### 2. AI Financial Assistant (`ai_assistant.py`)
Chatbot interface that answers common investment questions about:
- SIP benefits and strategies
- Risk management techniques
- Portfolio diversification
- Tax-saving investments
- Retirement planning

### 3. AI-Powered Risk Assessment (`ai_risk_assessment.py`)
Uses machine learning to analyze user behavior patterns and refine risk profiling:
- **User Behavior Analyzer**: Analyzes user responses and profile data to predict risk tolerance
- **User Cluster Analyzer**: Groups users with similar profiles for better recommendations
- **Enhanced Risk Profiler**: Combines traditional and AI-based risk profiling for more accurate assessments

### 4. Manual Training Script (`train_ai_models.py`)
Allows manual training of AI models with custom data:
- Train behavior analysis model with Random Forest Classifier
- Train user clustering model with K-Means
- Save trained models for later use
- Test models with new user profiles

### 5. Kaggle Training (`kaggle_risk_assessment.ipynb`)
Train AI models using Kaggle's powerful computing resources:
- Jupyter notebook optimized for Kaggle environment
- Synthetic data generation for training
- Comprehensive visualization and evaluation
- Model export for application integration

## Usage

1. **Risk Profiling**: Complete the interactive questionnaire to determine your risk profile
2. **Portfolio Generation**: Get AI-enhanced fund recommendations
3. **Backtesting**: Validate your strategy with historical data
4. **What-If Scenarios**: Model different market conditions
5. **AI Assistant**: Ask investment questions anytime

## Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Plotly for interactive charts
- **Machine Learning**: Scikit-learn for predictive modeling
- **Data Storage**: CSV files for fund data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.