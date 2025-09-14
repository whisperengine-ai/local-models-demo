#!/usr/bin/env python3
"""
context_window_demo.py

Comprehensive demonstration of context window management in the air-gapped LLM system.
Shows how to control input/output length, manage conversations, and handle token limits.

Usage: python context_window_demo.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

class ContextWindowManager:
    """Manages context window for conversations and text generation"""
    
    def __init__(self, model_path="./local_models/chat_model", max_context_tokens=800):
        """
        Initialize the context window manager
        
        Args:
            model_path: Path to the local model
            max_context_tokens: Maximum tokens to keep in context (leave room for response)
        """
        self.max_context_tokens = max_context_tokens
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"🎯 Loading model on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        
        if self.device != "cpu":
            self.model = self.model.to(torch.device(self.device))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
    
    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def truncate_context(self, text, max_tokens):
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Keep the most recent tokens
        truncated_tokens = tokens[-max_tokens:]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def generate_with_context_control(self, prompt, max_new_tokens=50, temperature=0.8):
        """Generate response with context window management"""
        
        # Ensure prompt fits in context window (leave room for response)
        available_tokens = self.max_context_tokens - max_new_tokens
        prompt = self.truncate_context(prompt, available_tokens)
        
        # Tokenize and move to device
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(torch.device(self.device))
        
        print(f"📏 Input tokens: {inputs.shape[1]}, Max new: {max_new_tokens}")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new part
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

def demo_context_window_scenarios():
    """Demonstrate various context window management scenarios"""
    
    manager = ContextWindowManager()
    
    print("\n" + "="*60)
    print("🪟 Context Window Management Demonstrations")
    print("="*60)
    
    # 1. Basic length control
    print("\n1️⃣ Response Length Control:")
    prompt = "Explain artificial intelligence"
    
    for length in [20, 50, 100]:
        response = manager.generate_with_context_control(prompt, max_new_tokens=length)
        print(f"   🎯 {length} tokens: \"{response}\"")
    
    # 2. Long input handling
    print("\n2️⃣ Long Input Handling:")
    long_text = """
    Artificial intelligence represents a transformative technology that is reshaping industries
    across the globe. From healthcare to finance, from transportation to entertainment,
    AI systems are becoming increasingly sophisticated and capable of performing tasks
    that were once thought to be exclusively human. Machine learning algorithms can now
    process vast amounts of data, identify patterns, and make predictions with remarkable
    accuracy. Deep learning networks have revolutionized computer vision, natural language
    processing, and speech recognition. The implications of these advances are profound
    and far-reaching, affecting how we work, live, and interact with technology.
    """
    
    token_count = manager.count_tokens(long_text)
    print(f"   📝 Original text: {token_count} tokens")
    
    response = manager.generate_with_context_control(long_text, max_new_tokens=40)
    print(f"   🤖 Response: \"{response}\"")
    
    # 3. Conversation context management
    print("\n3️⃣ Conversation Context Management:")
    
    class ConversationManager:
        def __init__(self, context_manager):
            self.context_manager = context_manager
            self.conversation_history = ""
        
        def add_turn(self, user_msg, bot_response):
            turn = f"User: {user_msg}\nBot: {bot_response}\n"
            self.conversation_history += turn
            
            # Check if we need to truncate
            tokens = self.context_manager.count_tokens(self.conversation_history)
            if tokens > self.context_manager.max_context_tokens - 100:  # Leave room for response
                print(f"   ⚠️  Truncating conversation (was {tokens} tokens)")
                self.conversation_history = self.context_manager.truncate_context(
                    self.conversation_history, 
                    self.context_manager.max_context_tokens - 200
                )
        
        def chat(self, user_msg):
            full_context = self.conversation_history + f"User: {user_msg}\nBot:"
            response = self.context_manager.generate_with_context_control(
                full_context, 
                max_new_tokens=50
            )
            self.add_turn(user_msg, response)
            return response
    
    conv = ConversationManager(manager)
    
    conversation = [
        "Hello, how are you?",
        "What do you think about AI?",
        "Can you explain machine learning?",
        "What are the benefits of automation?",
        "How does natural language processing work?",
        "What's the future of AI technology?"
    ]
    
    for msg in conversation:
        response = conv.chat(msg)
        print(f"   💬 User: {msg}")
        print(f"   🤖 Bot: {response}")
        
        # Show context size
        context_tokens = manager.count_tokens(conv.conversation_history)
        print(f"   📊 Context: {context_tokens} tokens")
        print()
    
    # 4. Token efficiency analysis
    print("4️⃣ Token Efficiency Analysis:")
    
    test_phrases = [
        "Hello",
        "How are you doing today?",
        "Can you explain the concept of machine learning in simple terms?",
        "I would like to understand how artificial intelligence systems work and what makes them so powerful in processing information"
    ]
    
    for phrase in test_phrases:
        tokens = manager.count_tokens(phrase)
        chars = len(phrase)
        ratio = chars / tokens if tokens > 0 else 0
        print(f"   📝 \"{phrase}\"")
        print(f"      🔢 {tokens} tokens, {chars} chars (ratio: {ratio:.1f} chars/token)")
    
    print(f"\n💡 Context Window Best Practices:")
    print(f"   • DialoGPT-medium max: ~1024 tokens")
    print(f"   • Reserve 200-300 tokens for responses")
    print(f"   • 1 token ≈ 0.75 words (English)")
    print(f"   • Monitor conversation length in long chats")
    print(f"   • Truncate old messages when near limit")
    print(f"   • Use max_new_tokens to control response length")

if __name__ == "__main__":
    print("🚀 Context Window Management Demo")
    demo_context_window_scenarios()
    print("\n✅ Demo complete! Use these patterns in your applications.")