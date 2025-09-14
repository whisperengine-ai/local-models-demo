#!/usr/bin/env python3
"""
example_llm_usage.py

Example script showing how to use the air-gapped LLM system with custom parameters.
This demonstrates practical usage patterns for different scenarios.

Usage: python example_llm_usage.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from offline_utils import enforce_offline_mode, get_safe_model_loading_kwargs, check_local_model_exists

# Set offline mode using standardized enforcement
enforce_offline_mode()

class LLMGenerator:
    """Simple wrapper for the air-gapped LLM with parameter control"""
    
    def __init__(self, model_path="./local_models/chat_model"):
        """Initialize the LLM generator"""
        print("🔄 Loading model...")
        
        # Determine best device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"🎯 Using device: {self.device}")
        
        # Check model exists locally
        if not check_local_model_exists(model_path):
            raise RuntimeError(f"Model not found at {model_path}")
        
        # Load tokenizer and model with standardized safe parameters
        safe_kwargs = get_safe_model_loading_kwargs()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **safe_kwargs)
        
        model_kwargs = safe_kwargs.copy()
        model_kwargs['torch_dtype'] = torch.float16 if self.device != "cpu" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if self.device != "cpu":
            self.model = self.model.to(torch.device(self.device))
        
        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
    
    def generate(self, prompt, 
                 temperature=0.8,
                 max_new_tokens=50,
                 top_p=0.9,
                 top_k=50,
                 repetition_penalty=1.1):
        """
        Generate response with custom parameters
        
        Args:
            prompt (str): Input text
            temperature (float): Randomness (0.1=focused, 2.0=creative)
            max_new_tokens (int): Maximum tokens to generate
            top_p (float): Nucleus sampling threshold
            top_k (int): Top-k sampling
            repetition_penalty (float): Penalty for repetition
        
        Returns:
            str: Generated response
        """
        # Tokenize input
        inputs = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
        inputs = inputs.to(torch.device(self.device))
        
        # Generate response
        with torch.no_grad():
            attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(torch.device(self.device))
            
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response (skip input)
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

def demo_scenarios():
    """Demonstrate different usage scenarios"""
    
    # Initialize generator
    llm = LLMGenerator()
    
    print("\n" + "="*60)
    print("🎭 LLM Parameter Control Demo")
    print("="*60)
    
    scenarios = [
        {
            "name": "📚 Factual Q&A",
            "prompt": "What is machine learning?",
            "params": {"temperature": 0.3, "max_new_tokens": 40, "top_p": 0.7},
            "description": "Low temperature for focused, factual responses"
        },
        {
            "name": "🎨 Creative Writing",
            "prompt": "Once upon a time in a magical forest,",
            "params": {"temperature": 1.2, "max_new_tokens": 80, "top_p": 0.95, "top_k": 100},
            "description": "High temperature for creative, diverse responses"
        },
        {
            "name": "💼 Professional Tone",
            "prompt": "Please explain the benefits of automation",
            "params": {"temperature": 0.5, "max_new_tokens": 60, "repetition_penalty": 1.3},
            "description": "Moderate settings for professional communication"
        },
        {
            "name": "🗨️ Casual Chat",
            "prompt": "How was your day?",
            "params": {"temperature": 0.8, "max_new_tokens": 30, "top_p": 0.9},
            "description": "Balanced settings for natural conversation"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print(f"⚙️ Parameters: {scenario['params']}")
        print(f"💬 Prompt: \"{scenario['prompt']}\"")
        
        response = llm.generate(scenario['prompt'], **scenario['params'])
        print(f"🤖 Response: \"{response}\"")
        print("-" * 50)
    
    print("\n🎯 Context Window Control Demo")
    print("-" * 40)
    
    base_prompt = "Artificial intelligence is transforming"
    
    for tokens in [20, 50, 100]:
        response = llm.generate(
            base_prompt, 
            temperature=0.7, 
            max_new_tokens=tokens
        )
        print(f"📏 {tokens} tokens: \"{response}\"")
    
    print("\n✅ Demo complete! You can now use these patterns in your applications.")

def interactive_mode():
    """Interactive chat mode with parameter control"""
    print("\n🤖 Interactive Chat Mode")
    print("Type 'quit' to exit, 'params' to change settings")
    print("-" * 50)
    
    llm = LLMGenerator()
    
    # Default parameters
    params = {
        "temperature": 0.8,
        "max_new_tokens": 50,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'params':
            print(f"\nCurrent parameters: {params}")
            print("Enter new values (press Enter to keep current):")
            
            for key in params:
                current = params[key]
                new_val = input(f"{key} (current: {current}): ").strip()
                if new_val:
                    try:
                        params[key] = float(new_val) if '.' in new_val else int(new_val)
                    except ValueError:
                        print(f"Invalid value for {key}, keeping {current}")
            continue
        
        if user_input:
            response = llm.generate(user_input, **params)
            print(f"🤖 Bot: {response}")

if __name__ == "__main__":
    print("🚀 Air-Gapped LLM Usage Examples")
    print("Choose a mode:")
    print("1. Demo scenarios")
    print("2. Interactive chat")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_scenarios()
    elif choice == "2":
        interactive_mode()
    else:
        print("Running demo scenarios by default...")
        demo_scenarios()