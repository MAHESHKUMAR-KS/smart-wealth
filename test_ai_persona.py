"""
Test script for AI Risk Persona Engine
"""

from ai_risk_persona import get_investor_persona

def test_ai_persona():
    print("Testing AI Risk Persona Engine...")
    
    # Test case 1: Conservative investor
    print("\n--- Test Case 1: Conservative Investor ---")
    persona1 = get_investor_persona(
        age=55,
        horizon=10,
        risk_reaction="very_concerned",
        goal="capital_preservation",
        volatility="uncomfortable"
    )
    print(f"Persona: {persona1['name']}")
    print(f"Explanation: {persona1['explanation']}")
    print(f"Volatility Tolerance: {persona1['volatility_tolerance']}")
    equity_range = persona1['allocation_range']['equity']
    print(f"Equity Allocation Range: {equity_range[0]}-{equity_range[1]}%")
    
    # Test case 2: Aggressive investor
    print("\n--- Test Case 2: Aggressive Investor ---")
    persona2 = get_investor_persona(
        age=30,
        horizon=25,
        risk_reaction="very_comfortable",
        goal="aggressive_growth",
        volatility="very_comfortable"
    )
    print(f"Persona: {persona2['name']}")
    print(f"Explanation: {persona2['explanation']}")
    print(f"Volatility Tolerance: {persona2['volatility_tolerance']}")
    equity_range = persona2['allocation_range']['equity']
    print(f"Equity Allocation Range: {equity_range[0]}-{equity_range[1]}%")
    
    # Test case 3: Moderate investor
    print("\n--- Test Case 3: Moderate Investor ---")
    persona3 = get_investor_persona(
        age=40,
        horizon=15,
        risk_reaction="neutral",
        goal="balanced_growth",
        volatility="neutral"
    )
    print(f"Persona: {persona3['name']}")
    print(f"Explanation: {persona3['explanation']}")
    print(f"Volatility Tolerance: {persona3['volatility_tolerance']}")
    equity_range = persona3['allocation_range']['equity']
    print(f"Equity Allocation Range: {equity_range[0]}-{equity_range[1]}%")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_ai_persona()