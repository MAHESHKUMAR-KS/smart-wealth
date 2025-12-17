"""
Debug version of WealthyWise Assistant to help troubleshoot real-time functionality
"""

import streamlit as st
from typing import Dict, Any


class WealthyWiseAssistant:
    """Professional investment education chatbot"""
    
    def __init__(self):
        """Initialize the assistant with educational responses"""
        self.persona_descriptions = {
            'Capital Preservation Planner': {
                'focus': 'protecting principal investment with minimal risk exposure',
                'characteristics': [
                    'Prioritizes safety over returns',
                    'Prefers stable, predictable investments',
                    'Typically has shorter investment horizons',
                    'Concerned about market volatility'
                ]
            },
            'Income-Focused Stabilizer': {
                'focus': 'generating regular income while maintaining stability',
                'characteristics': [
                    'Seeks balance between income and stability',
                    'Moderate risk tolerance',
                    'Interested in consistent returns',
                    'Values regular cash flows'
                ]
            },
            'Long-Term Growth Optimizer': {
                'focus': 'achieving growth over extended periods with moderate volatility',
                'characteristics': [
                    'Balanced approach to risk and return',
                    'Longer investment time horizon',
                    'Accepts moderate fluctuations for growth',
                    'Focused on wealth building over time'
                ]
            },
            'Opportunistic Risk Taker': {
                'focus': 'maximizing growth potential with higher risk acceptance',
                'characteristics': [
                    'High tolerance for market volatility',
                    'Seeking maximum growth opportunities',
                    'Long investment time horizon',
                    'Comfortable with significant portfolio swings'
                ]
            }
        }
    
    def explain_risk_persona(self, persona_name: str, persona_details: Dict) -> str:
        """Explain the user's risk persona in educational terms"""
        if persona_name not in self.persona_descriptions:
            return "I can help explain your investor persona once it's determined."
        
        desc = self.persona_descriptions[persona_name]
        explanation = f"As a **{persona_name}**, your investment approach focuses on {desc['focus']}. "
        explanation += "This means:\n\n"
        
        for characteristic in desc['characteristics']:
            explanation += f"- {characteristic}\n"
        
        explanation += f"\nThis persona typically aligns with an equity allocation range of "
        equity_range = persona_details['allocation_range']['equity']
        explanation += f"{equity_range[0]}-{equity_range[1]}%."
        
        return explanation
    
    def explain_asset_allocation(self, equity_pct: float, debt_pct: float, persona_name: str) -> str:
        """Explain asset allocation concepts"""
        explanation = f"Your portfolio has {equity_pct}% in growth-oriented investments and {debt_pct}% in stability-focused investments.\n\n"
        
        if equity_pct <= 20:
            explanation += "This conservative allocation prioritizes capital preservation over growth, suitable for short-term goals or risk-averse investors."
        elif equity_pct <= 40:
            explanation += "This moderate allocation balances income generation with some growth potential, appropriate for investors seeking stability with modest growth."
        elif equity_pct <= 70:
            explanation += "This balanced allocation aims for growth while maintaining some stability, fitting investors with medium-term horizons and moderate risk tolerance."
        else:
            explanation += "This growth-oriented allocation emphasizes long-term appreciation potential, suited for investors with long horizons and high risk tolerance."
        
        return explanation
    
    def explain_volatility_tolerance(self, tolerance_level: str) -> str:
        """Explain volatility tolerance levels"""
        explanations = {
            'Low': "You prefer investments with minimal price fluctuations. This typically means focusing on bonds, fixed deposits, and stable value funds.",
            'Low-Medium': "You're comfortable with modest market movements. This usually involves a mix of bonds and blue-chip stocks with moderate volatility.",
            'Medium': "You accept moderate price swings for potential growth. This often includes diversified stock funds and balanced portfolios.",
            'High': "You're comfortable with significant market movements. This typically involves growth stocks, sector funds, and other higher-volatility investments."
        }
        
        return explanations.get(tolerance_level, "Your volatility tolerance level helps determine how much market fluctuation you can comfortably withstand.")
    
    def answer_question(self, question: str, user_context: Dict[str, Any]) -> str:
        """Answer user questions based on their context"""
        question = question.lower()
        
        # Debug information
        debug_info = f"\n\n**Debug Info:**\n"
        debug_info += f"- Persona available: {'Yes' if user_context.get('persona') else 'No'}\n"
        debug_info += f"- Portfolio available: {'Yes' if len(user_context.get('portfolio', [])) > 0 else 'No'}\n"
        debug_info += f"- User profile available: {'Yes' if user_context.get('user_profile') else 'No'}\n"
        
        # Handle questions about risk persona
        if any(word in question for word in ['risk', 'persona', 'type', 'profile']):
            if 'persona' in user_context and user_context['persona']:
                persona_name = user_context['persona']['name']
                response = self.explain_risk_persona(persona_name, user_context['persona'])
                return response + debug_info
            else:
                return "Your investor persona is determined based on your age, investment timeline, and how you react to market changes. This helps create a personalized approach to investing that matches your comfort level and goals." + debug_info
        
        # Handle questions about asset allocation
        elif any(word in question for word in ['allocation', 'equity', 'debt', 'split', 'distribution']):
            if 'portfolio' in user_context and len(user_context['portfolio']) > 0:
                # Simplified calculation for explanation
                equity_pct = 60  # Placeholder - would be calculated from actual portfolio
                debt_pct = 40    # Placeholder - would be calculated from actual portfolio
                persona_name = user_context.get('persona', {}).get('name', '')
                response = self.explain_asset_allocation(equity_pct, debt_pct, persona_name)
                return response + debug_info
            else:
                return "Asset allocation refers to how your investments are divided between growth-focused assets (like stocks) and stability-focused assets (like bonds). The right mix depends on your goals, timeline, and comfort with market ups and downs." + debug_info
        
        # Handle questions about volatility
        elif any(word in question for word in ['volatility', 'fluctuation', 'swing', 'movement']):
            tolerance = user_context.get('persona', {}).get('volatility_tolerance', 'Medium')
            response = self.explain_volatility_tolerance(tolerance)
            return response + debug_info
        
        # Handle questions about time horizon
        elif any(word in question for word in ['horizon', 'time', 'long', 'short']):
            horizon = user_context.get('user_profile', {}).get('horizon', 10)
            if horizon <= 3:
                response = "With a short investment horizon, preserving capital becomes more important than maximizing growth, as there's less time to recover from potential losses."
            elif horizon <= 7:
                response = "A medium-term horizon allows for some market exposure while still requiring stability, balancing growth potential with risk management."
            else:
                response = "A long investment horizon provides flexibility to pursue growth-oriented investments, as you have time to ride out market fluctuations."
            return response + debug_info
        
        # Handle questions about returns/guarantees
        elif any(word in question for word in ['return', 'guarantee', 'performance', 'predict']):
            response = "It's important to understand that all investments carry risks, and past performance doesn't guarantee future results. The goal is to find an approach that matches your risk tolerance while pursuing your financial objectives."
            return response + debug_info
        
        # Handle questions about specific investments
        elif any(word in question for word in ['fund', 'stock', 'bond', 'buy', 'sell', 'invest']):
            response = "I can help you understand factors to consider when evaluating investments, such as fees, diversification, and alignment with your goals. However, I don't recommend specific securities. Focus on building a well-balanced portfolio that matches your risk profile and timeline."
            return response + debug_info
        
        # Handle general investment questions
        elif any(word in question for word in ['diversification', 'diversify']):
            response = "Diversification means spreading investments across different asset types, sectors, and geographic regions. This helps reduce risk by ensuring that poor performance in one area doesn't devastate your entire portfolio."
            return response + debug_info
        
        # Default response
        else:
            response = "I'm here to help you understand investment concepts and your personalized portfolio approach. Could you ask a more specific question about your investor persona, asset allocation, or investment strategy?"
            return response + debug_info
    
    def get_educational_takeaway(self, question: str) -> str:
        """Provide a short educational takeaway"""
        takeaways = {
            'risk': "Remember: Risk and return are inherently linked in investing - higher potential returns typically require accepting higher uncertainty.",
            'allocation': "Key insight: Asset allocation has a greater impact on portfolio outcomes than individual security selection.",
            'volatility': "Important concept: Volatility isn't inherently good or bad - it's about finding the right level for your comfort and goals.",
            'horizon': "Valuable principle: Time is often an investor's greatest ally, allowing recovery from short-term market downturns.",
            'diversification': "Essential strategy: Don't put all your eggs in one basket - diversification helps smooth out investment returns.",
            'default': "Investment wisdom: Understanding your own risk tolerance and investment timeline is the foundation of sound financial planning."
        }
        
        question = question.lower()
        for key, takeaway in takeaways.items():
            if key in question:
                return takeaway
        
        return takeaways['default']


def render_debug_chat_interface():
    """Render the debug chat interface in Streamlit"""
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'assistant' not in st.session_state:
        st.session_state.assistant = WealthyWiseAssistant()
    
    st.header("🎓 WealthyWise Assistant (Debug Mode)")
    st.caption("Your investment education companion with debugging info")
    
    # Display current session state info
    st.subheader("Current Session State")
    st.write(f"User persona available: {'Yes' if st.session_state.get('user_persona') else 'No'}")
    st.write(f"Recommended portfolio available: {'Yes' if st.session_state.get('recommended_portfolio') is not None else 'No'}")
    st.write(f"User profile available: {'Yes' if st.session_state.get('user_profile') else 'No'}")
    
    if st.session_state.get('user_persona'):
        st.json(st.session_state.user_persona)
    
    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about your investor persona or portfolio..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        user_context = {
            'persona': st.session_state.get('user_persona', {}),
            'portfolio': st.session_state.get('recommended_portfolio', []),
            'user_profile': st.session_state.get('user_profile', {})
        }
        
        response = st.session_state.assistant.answer_question(prompt, user_context)
        takeaway = st.session_state.assistant.get_educational_takeaway(prompt)
        full_response = f"{response}\n\n**Educational takeaway:** {takeaway}"
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(full_response)


if __name__ == "__main__":
    # This would run if the file is executed directly
    st.set_page_config(page_title="WealthyWise Assistant Debug", page_icon="🎓")
    render_debug_chat_interface()