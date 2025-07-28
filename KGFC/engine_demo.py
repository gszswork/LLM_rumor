#!/usr/bin/env python3
"""
Demonstration of the updated run_llm function with GPT and Gemini support
"""

from fc_helpers import run_llm

def demo_engines():
    test_prompt = "Explain what 2+2 equals in one sentence."
    
    print("=== LLM Engine Demonstration ===\n")
    
    # Test GPT engines
    print("1. GPT Models (requires OPENAI_API_KEY):")
    gpt_engines = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    for engine in gpt_engines:
        print(f"   Engine: {engine}")
        result = run_llm(test_prompt, engine=engine, max_tokens=50)
        print(f"   Result: {result[:100]}")
        print()
    
    # Test Gemini engines  
    print("2. Gemini Models (requires GOOGLE_API_KEY or GEMINI_API_KEY):")
    gemini_engines = ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    for engine in gemini_engines:
        print(f"   Engine: {engine}")
        result = run_llm(test_prompt, engine=engine, max_tokens=50)
        print(f"   Result: {result[:100]}")
        print()
    
    print("3. Unsupported Engine (falls back to mock):")
    result = run_llm(test_prompt, engine="unsupported-model", max_tokens=50)
    print(f"   Result: {result[:100]}")
    
    print("\n=== Setup Instructions ===")
    print("To use real LLMs, set environment variables:")
    print("  export OPENAI_API_KEY='your-openai-key'")
    print("  export GOOGLE_API_KEY='your-google-key'")
    print("  # or export GEMINI_API_KEY='your-gemini-key'")

if __name__ == "__main__":
    demo_engines()