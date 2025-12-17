"""
AI Chatbot Module for WealthyWise Assistant

This module provides educational responses about investor personas and portfolios
without giving specific investment advice.
"""

def get_chatbot_response(user_question: str, persona_data: dict) -> str:
    """
    Generate an educational response based on user question and persona data.
    
    Args:
        user_question (str): The user's question
        persona_data (dict): The user's persona data from st.session_state
        
    Returns:
        str: Educational response about investing concepts
    """
    # Convert question to lowercase for easier matching
    question = user_question.lower()
    
    # Handle questions about risk persona
    if any(keyword in question for keyword in ['persona', 'risk', 'type', 'profile']):
        if not persona_data:
            return "Please generate your risk profile first to get personalized insights about your investor persona."
        
        persona_name = persona_data.get('name', 'Unknown')
        explanation = persona_data.get('explanation', '')
        volatility = persona_data.get('volatility_tolerance', 'Unknown')
        
        equity_range = persona_data.get('allocation_range', {}).get('equity', [0, 0])
        debt_range = persona_data.get('allocation_range', {}).get('debt', [0, 0])
        
        response = f"As a **{persona_name}**, your investment approach is designed around {explanation.lower()}. "
        response += f"This means you have a **{volatility}** tolerance for market fluctuations.\n\n"
        response += f"Based on this persona, a typical allocation would be:\n"
        response += f"- **Equity**: {equity_range[0]}-{equity_range[1]}%\n"
        response += f"- **Debt**: {debt_range[0]}-{debt_range[1]}%\n\n"
        response += "This allocation balances your growth potential with your comfort level regarding market volatility."
        
        return response
    
    # Handle questions about allocation
    elif any(keyword in question for keyword in ['allocation', 'equity', 'debt', 'split', 'distribution']):
        if not persona_data:
            return "Please generate your risk profile first to get personalized insights about asset allocation."
        
        equity_range = persona_data.get('allocation_range', {}).get('equity', [0, 0])
        debt_range = persona_data.get('allocation_range', {}).get('debt', [0, 0])
        
        response = f"Your recommended asset allocation is:\n"
        response += f"- **Equity (Growth)**: {equity_range[0]}-{equity_range[1]}%\n"
        response += f"- **Debt (Stability)**: {debt_range[0]}-{debt_range[1]}%\n\n"
        
        if equity_range[1] <= 20:
            response += "This conservative allocation prioritizes capital preservation over growth, suitable for short-term goals or risk-averse investors."
        elif equity_range[1] <= 40:
            response += "This moderate allocation balances income generation with some growth potential, appropriate for investors seeking stability with modest growth."
        elif equity_range[1] <= 70:
            response += "This balanced allocation aims for growth while maintaining some stability, fitting investors with medium-term horizons and moderate risk tolerance."
        else:
            response += "This growth-oriented allocation emphasizes long-term appreciation potential, suited for investors with long horizons and high risk tolerance."
        
        return response
    
    # Handle questions about volatility
    elif any(keyword in question for keyword in ['volatility', 'fluctuation', 'swing', 'movement']):
        if not persona_data:
            return "Please generate your risk profile first to get personalized insights about volatility tolerance."
        
        volatility = persona_data.get('volatility_tolerance', 'Unknown')
        
        response = f"Your volatility tolerance level is classified as **{volatility}**.\n\n"
        
        if volatility == 'Low':
            response += "You prefer investments with minimal price fluctuations. This typically means focusing on bonds, fixed deposits, and stable value funds."
        elif volatility == 'Low-Medium':
            response += "You're comfortable with modest market movements. This usually involves a mix of bonds and blue-chip stocks with moderate volatility."
        elif volatility == 'Medium':
            response += "You accept moderate price swings for potential growth. This often includes diversified stock funds and balanced portfolios."
        elif volatility == 'High':
            response += "You're comfortable with significant market movements. This typically involves growth stocks, sector funds, and other higher-volatility investments."
        else:
            response += "This indicates your comfort level with market ups and downs, which helps determine suitable investment choices."
        
        return response
    
    # Handle questions about time horizon
    elif any(keyword in question for keyword in ['horizon', 'time', 'long', 'short']):
        if not persona_data:
            return "Please generate your risk profile first to get personalized insights about investment horizons."
        
        # This would ideally come from user profile data
        response = "Your investment horizon is an important factor that influences your portfolio construction:\n\n"
        response += "- **Short-term (< 3 years)**: Focus on capital preservation\n"
        response += "- **Medium-term (3-7 years)**: Balance growth and stability\n"
        response += "- **Long-term (> 7 years)**: Emphasize growth potential\n\n"
        response += "Generally, longer horizons allow for more growth-oriented investments since you have time to recover from market downturns."
        
        return response
    
    # Handle questions about diversification
    elif 'diversification' in question or 'diversify' in question:
        response = "Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio. "
        response += "The rationale is that different asset types, sectors, and geographic regions rarely all perform poorly at the same time. "
        response += "By spreading investments across various categories, you can potentially reduce the impact of any single poor-performing investment on your overall portfolio.\n\n"
        response += "Key principles include:\n"
        response += "- Spread investments across different asset classes (stocks, bonds, etc.)\n"
        response += "- Invest in various sectors and industries\n"
        response += "- Consider different geographic regions\n"
        response += "- Include various market capitalizations (large-cap, mid-cap, small-cap)\n\n"
        response += "Remember: Don't put all your eggs in one basket!"
        
        return response
    
    # Handle questions about returns/performance
    elif any(keyword in question for keyword in ['return', 'performance', 'gain', 'profit']):
        response = "It's important to understand that all investments carry risks, and past performance doesn't guarantee future results. "
        response += "Key concepts to consider:\n\n"
        response += "- **Risk and Return Relationship**: Higher potential returns typically require accepting higher uncertainty\n"
        response += "- **Time Horizon**: Longer investment periods generally allow for more growth-oriented strategies\n"
        response += "- **Diversification**: Spreading investments can help smooth out returns over time\n"
        response += "- **Costs Matter**: Fees and expenses can significantly impact long-term returns\n\n"
        response += "Focus on building a well-balanced portfolio that matches your risk profile and timeline rather than chasing specific returns."
        
        return response
    
    # Handle questions about specific investments
    elif any(keyword in question for keyword in ['fund', 'stock', 'bond', 'buy', 'sell', 'invest']):
        response = "I can help you understand factors to consider when evaluating investments, such as fees, diversification, and alignment with your goals. "
        response += "However, I don't recommend specific securities or provide buy/sell advice.\n\n"
        response += "When evaluating potential investments, consider:\n"
        response += "- Alignment with your risk tolerance and investment timeline\n"
        response += "- Fees and expense ratios\n"
        response += "- Historical performance relative to benchmarks\n"
        response += "- Portfolio diversification benefits\n"
        response += "- Fund manager experience and strategy\n\n"
        response += "Always consult with a qualified financial advisor for personalized investment advice."
        
        return response
    
    # Default response for unrecognized questions
    else:
        response = "I'm here to help you understand investment concepts and your personalized portfolio approach. "
        response += "I can explain your investor persona, asset allocation, volatility tolerance, diversification strategies, and other educational topics.\n\n"
        response += "Try asking questions like:\n"
        response += "- 'What does my investor persona mean?'\n"
        response += "- 'Why is my allocation this way?'\n"
        response += "- 'What is diversification?'\n"
        response += "- 'How much volatility can I handle?'\n\n"
        response += "Remember, I provide educational information only and do not offer specific investment advice."
        
        return response