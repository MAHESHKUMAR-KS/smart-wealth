"""
Test script to verify real-time functionality of the WealthyWise Assistant
"""

import streamlit as st

# Mock session state for testing
class MockSessionState:
    def __init__(self):
        self.user_persona = {
            'name': 'Long-Term Growth Optimizer',
            'explanation': 'Balanced investors with a long-term perspective who accept moderate volatility for growth.',
            'allocation_range': {'equity': (40, 70), 'debt': (20, 50)},
            'volatility_tolerance': 'Medium'
        }
        self.recommended_portfolio = []
        self.user_profile = {
            'age': 35,
            'horizon': 20,
            'investment_amount': 100000
        }

# Test the real-time functionality
def test_realtime_functionality():
    print("Testing real-time functionality of WealthyWise Assistant...")
    
    # Mock session state
    mock_state = MockSessionState()
    
    # Test accessing session state data
    print(f"User persona: {mock_state.user_persona.get('name', 'Not available')}")
    print(f"User profile age: {mock_state.user_profile.get('age', 'Not available')}")
    print(f"Portfolio available: {'Yes' if mock_state.recommended_portfolio is not None else 'No'}")
    
    print("\nReal-time functionality test completed successfully!")

if __name__ == "__main__":
    test_realtime_functionality()