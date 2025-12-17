"""
Test script for the WealthyWise Assistant chatbot
"""

from ai_chatbot import get_chatbot_response

def test_chatbot():
    # Sample persona data
    sample_persona = {
        'name': 'Balanced Growth Seeker',
        'explanation': 'Moderate-risk investors seeking a balance between steady growth and capital protection.',
        'allocation_range': {'equity': (30, 60), 'debt': (40, 70)},
        'volatility_tolerance': 'Medium'
    }
    
    # Test questions
    questions = [
        "What is my investor persona?",
        "Why is my allocation this way?",
        "What does volatility tolerance mean?",
        "How does my investment horizon affect my portfolio?",
        "What is diversification?",
        "Can you predict my returns?",
        "Should I buy this specific fund?",
        "What does my risk profile mean?"
    ]
    
    print("Testing WealthyWise Assistant Chatbot")
    print("=" * 50)
    
    for question in questions:
        print(f"\nQ: {question}")
        response = get_chatbot_response(question, sample_persona)
        print(f"A: {response}")
        print("-" * 30)
    
    # Test with no persona data
    print("\nTesting with no persona data:")
    print("=" * 30)
    response = get_chatbot_response("What is my persona?", None)
    print(f"Response: {response}")

if __name__ == "__main__":
    test_chatbot()