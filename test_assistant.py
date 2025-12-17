"""
Test script for WealthyWise Assistant
"""

from wealthywise_assistant import WealthyWiseAssistant

def test_assistant():
    print("Testing WealthyWise Assistant...")
    
    # Initialize assistant
    assistant = WealthyWiseAssistant()
    
    # Test context
    test_context = {
        'persona': {
            'name': 'Long-Term Growth Optimizer',
            'explanation': 'Balanced investors with a long-term perspective who accept moderate volatility for growth.',
            'allocation_range': {'equity': (40, 70), 'debt': (20, 50)},
            'volatility_tolerance': 'Medium'
        },
        'user_profile': {
            'age': 35,
            'horizon': 20,
            'investment_amount': 100000
        }
    }
    
    # Test questions
    questions = [
        "Why is my risk profile this way?",
        "What does my asset allocation mean?",
        "How much volatility can I handle?",
        "Is my investment timeline appropriate?",
        "Should I buy this specific fund?",
        "What is diversification?"
    ]
    
    print("\n--- Testing Assistant Responses ---")
    for question in questions:
        print(f"\nQ: {question}")
        response = assistant.answer_question(question, test_context)
        print(f"A: {response}")
        
        takeaway = assistant.get_educational_takeaway(question)
        print(f"Takeaway: {takeaway}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_assistant()