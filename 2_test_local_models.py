#!/usr/bin/env python3
"""
2_test_local_models.py

Run this script to test your locally downloaded models.
This script works completely offline - no internet required!

Usage: python 2_test_local_models.py
"""

import os
import sys
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from offline_utils import enforce_offline_mode, get_safe_model_loading_kwargs, check_local_model_exists

class LLMConfig:
    """Configuration class for controlling LLM generation parameters"""
    
    def __init__(self, 
                 temperature=0.8,
                 max_new_tokens=50,
                 top_p=0.9,
                 top_k=50,
                 repetition_penalty=1.1,
                 do_sample=True,
                 num_return_sequences=1):
        """
        Initialize LLM configuration parameters
        
        Args:
            temperature (float): Controls randomness (0.1-2.0). Lower = more focused, Higher = more creative
            max_new_tokens (int): Maximum number of new tokens to generate (context window extension)
            top_p (float): Nucleus sampling - probability mass to consider (0.1-1.0)
            top_k (int): Top-k sampling - number of highest probability tokens to consider
            repetition_penalty (float): Penalty for repeating tokens (1.0-2.0)
            do_sample (bool): Whether to use sampling (True) or greedy decoding (False)
            num_return_sequences (int): Number of different responses to generate
        """
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
    
    def to_dict(self):
        """Convert config to dictionary for model.generate()"""
        return {
            'temperature': self.temperature,
            'max_new_tokens': self.max_new_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'do_sample': self.do_sample,
            'num_return_sequences': self.num_return_sequences
        }
    
    @classmethod
    def creative(cls):
        """High creativity configuration"""
        return cls(temperature=1.2, max_new_tokens=100, top_p=0.95, top_k=100)
    
    @classmethod
    def balanced(cls):
        """Balanced configuration (default)"""
        return cls(temperature=0.8, max_new_tokens=50, top_p=0.9, top_k=50)
    
    @classmethod
    def focused(cls):
        """Low creativity, more focused responses"""
        return cls(temperature=0.3, max_new_tokens=30, top_p=0.7, top_k=20)
    
    @classmethod
    @classmethod
    def deterministic(cls):
        """Deterministic responses (no randomness)"""
        return cls(temperature=0.0, do_sample=False, max_new_tokens=50)

def get_best_device():
    """Determine the best available device for model inference"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def test_embedding_model():
    """Test local embedding model"""
    print("\n=== Testing Local Embedding Model ===")
    
    model_path = "./local_models/embedding_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please run '1_download_models.py' first")
        return
    
    try:
        # Determine best device
        device = get_best_device()
        print(f"🎯 Using device: {device}")
        
        # Load model from local filesystem
        model = SentenceTransformer(model_path, device=device)
        print("✅ Embedding model loaded successfully")
        
        # Test texts
        texts = [
            "The cat sits on the mat",
            "A feline rests on a rug",
            "Dogs love to play fetch",
            "Machine learning is fascinating"
        ]
        
        print("🔄 Generating embeddings...")
        embeddings = model.encode(texts)
        
        print(f"📊 Generated embeddings shape: {embeddings.shape}")
        print(f"📊 First embedding preview: [{embeddings[0][:3].round(3)}...]")
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        sim_cat_feline = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_cat_dog = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        
        print(f"🔍 Similarity (cat vs feline): {sim_cat_feline:.3f}")
        print(f"🔍 Similarity (cat vs dog): {sim_cat_dog:.3f}")
        print("✅ Embedding test completed successfully")
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")

def test_chat_model():
    """Test local chat model"""
    print("\n=== Testing Local Chat Model ===")
    
    model_path = "./local_models/chat_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please run '1_download_models.py' first")
        return
    
    try:
        # Determine best device
        device = get_best_device()
        print(f"🎯 Using device: {device}")
        
        # Check model exists locally
        if not check_local_model_exists(model_path):
            return
        
        # Load model with standardized safe parameters
        print("🔄 Loading tokenizer and model...")
        safe_kwargs = get_safe_model_loading_kwargs()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, **safe_kwargs)
        
        # Load model with device-specific optimizations
        model_kwargs = safe_kwargs.copy()
        model_kwargs['torch_dtype'] = torch.float16 if device != "cpu" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Move model to device if not CPU
        if device != "cpu":
            torch_device = torch.device(device)
            model = model.to(torch_device)
        
        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ Chat model loaded successfully on {device}")
        print(f"📍 Model device: {next(model.parameters()).device}")
        
        # Test different configurations
        configs = [
            ("🎯 Focused", LLMConfig.focused()),
            ("⚖️ Balanced", LLMConfig.balanced()),
            ("🎨 Creative", LLMConfig.creative())
        ]
        
        # Test prompts
        prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Tell me about machine learning"
        ]
        
        for config_name, config in configs:
            print(f"\n{config_name} Configuration:")
            print(f"   Temperature: {config.temperature}, Max tokens: {config.max_new_tokens}")
            print(f"   Top-p: {config.top_p}, Top-k: {config.top_k}")
            
            for prompt in prompts[:1]:  # Test with first prompt only for demo
                print(f"\n💬 Input: {prompt}")
                
                # Tokenize with proper attention mask
                inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
                
                # Move inputs to the same device as model
                device_obj = torch.device(device)
                inputs = inputs.to(device_obj)
                
                # Generate with custom configuration
                print("🔄 Generating response...")
                with torch.no_grad():
                    # Create attention mask
                    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device_obj)
                    
                    # Get config parameters
                    gen_params = config.to_dict()
                    
                    outputs = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_length=inputs.shape[1] + gen_params['max_new_tokens'],
                        num_return_sequences=gen_params['num_return_sequences'],
                        temperature=gen_params['temperature'],
                        top_p=gen_params['top_p'],
                        top_k=gen_params['top_k'],
                        do_sample=gen_params['do_sample'],
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=gen_params['repetition_penalty']
                    )
                
                # Decode only the generated part (skip the input)
                generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                print(f"🤖 Response: {generated_text}")
        
        # Demonstrate advanced context window control
        print(f"\n🪟 Advanced Context Window Control:")
        
        def demonstrate_context_control():
            """Show different context window management techniques"""
            
            # 1. Basic context window control
            print("\n1️⃣ Basic Context Window Control:")
            base_prompt = "Artificial intelligence is transforming industries"
            inputs = tokenizer.encode(base_prompt, return_tensors="pt").to(device_obj)
            print(f"   📏 Input tokens: {inputs.shape[1]}")
            
            for max_new in [10, 30, 80]:
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=max_new,  # This controls output length
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                print(f"   🎯 {max_new} new tokens: \"{response[:50]}{'...' if len(response) > 50 else ''}\"")
            
            # 2. Conversation context management
            print("\n2️⃣ Conversation Context Management:")
            conversation_history = ""
            turns = [
                "Hello, how are you?",
                "What's your favorite topic?",
                "Tell me more about that"
            ]
            
            for i, user_msg in enumerate(turns):
                # Build conversation context
                full_context = conversation_history + f"User: {user_msg}" + tokenizer.eos_token
                inputs = tokenizer.encode(full_context, return_tensors="pt").to(device_obj)
                
                # Limit context window if too long (prevent memory issues)
                max_context_tokens = 200  # Configurable context window limit
                if inputs.shape[1] > max_context_tokens:
                    inputs = inputs[:, -max_context_tokens:]  # Keep most recent tokens
                
                print(f"   Turn {i+1}: \"{user_msg}\" (Context: {inputs.shape[1]} tokens)")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=40,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                bot_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                print(f"   🤖 Response: \"{bot_response}\"")
                
                # Update conversation history
                conversation_history += f"User: {user_msg}{tokenizer.eos_token}Bot: {bot_response}{tokenizer.eos_token}"
            
            # 3. Token counting and limits
            print("\n3️⃣ Token Counting & Limits:")
            test_texts = [
                "Short text",
                "This is a medium length text that contains several words and should demonstrate token counting",
                "This is a very long text that goes on and on and demonstrates how we can measure the token count for different lengths of input text to understand context window management"
            ]
            
            for text in test_texts:
                tokens = tokenizer.encode(text)
                print(f"   📝 \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
                print(f"      🔢 Tokens: {len(tokens)} | Characters: {len(text)}")
            
            print(f"\n💡 Context Window Tips:")
            print(f"   • DialoGPT-medium max context: ~1024 tokens")
            print(f"   • 1 token ≈ 0.75 words (English)")
            print(f"   • Use max_new_tokens to control output length")
            print(f"   • Use max_length for total input+output limit")
            print(f"   • Truncate old conversation for long chats")
            print(f"   • Monitor token count to prevent memory issues")
        
        demonstrate_context_control()
        
        print("✅ Chat model test completed successfully")
        
    except Exception as e:
        print(f"❌ Chat model test failed: {e}")

def demo_custom_llm_parameters():
    """Demonstrate how to use custom LLM parameters in your own code"""
    print("\n=== Custom LLM Parameters Demo ===")
    print("Here's how to control LLM parameters in your own applications:")
    
    print("""
🎛️  Key Parameters Explained:

1. **Temperature** (0.1 - 2.0):
   • 0.1-0.3: Very focused, deterministic responses
   • 0.7-0.9: Balanced creativity and focus
   • 1.0-2.0: Very creative, more random responses

2. **Max New Tokens** (10 - 2048):
   • Controls length of generated response
   • Think of it as "context window extension"
   • Larger = longer responses, more memory usage

3. **Top-p** (0.1 - 1.0):
   • Nucleus sampling - probability mass to consider
   • 0.7: More focused vocabulary
   • 0.95: Wider vocabulary range

4. **Top-k** (1 - 100):
   • Number of highest probability tokens to consider
   • Lower = more focused, Higher = more diverse

5. **Repetition Penalty** (1.0 - 2.0):
   • 1.0: No penalty (may repeat)
   • 1.1-1.3: Good balance
   • 1.5+: Strong anti-repetition (may hurt coherence)

📝 Example Usage in Your Code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your models
tokenizer = AutoTokenizer.from_pretrained('./local_models/chat_model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('./local_models/chat_model', local_files_only=True)

# Create different configurations
creative_config = {
    'temperature': 1.2,
    'max_new_tokens': 100,
    'top_p': 0.95,
    'top_k': 100,
    'do_sample': True,
    'repetition_penalty': 1.1
}

focused_config = {
    'temperature': 0.3,
    'max_new_tokens': 50,
    'top_p': 0.7,
    'top_k': 20,
    'do_sample': True,
    'repetition_penalty': 1.2
}

# Generate with custom parameters
def generate_response(prompt, config):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, **config)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

# Use it
creative_response = generate_response("Tell me a story", creative_config)
focused_response = generate_response("What is AI?", focused_config)
```

🎯 Pro Tips:
• Start with balanced settings, then adjust based on your needs
• Lower temperature for factual Q&A, higher for creative writing
• Increase max_new_tokens for longer responses
• Use repetition_penalty > 1.0 to avoid repetitive outputs
• Test different combinations to find your sweet spot
    """)
    
    print("✅ Parameter explanation complete!")

def check_system_info():
    """Display comprehensive system information including GPU/Metal status"""
    print("\n=== System Information ===")
    
    # Python and PyTorch versions
    import sys
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    
    # CPU information
    print(f"🧠 CPU cores: {os.cpu_count()}")
    
    # GPU/Acceleration status
    print("\n--- Acceleration Hardware Status ---")
    
    # CUDA (NVIDIA GPUs)
    cuda_available = torch.cuda.is_available()
    print(f"💻 CUDA available: {cuda_available}")
    if cuda_available:
        print(f"🎮 CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"🎮 CUDA device count: {torch.cuda.device_count()}")
    
    # MPS (Apple Silicon Metal Performance Shaders)
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        print(f"🍎 MPS (Metal) available: {mps_available}")
        print(f"🍎 MPS (Metal) built: {mps_built}")
        
        if mps_available:
            print("✅ Apple Silicon GPU acceleration enabled")
            # Test MPS device
            try:
                device = torch.device("mps")
                test_tensor = torch.tensor([1.0, 2.0]).to(device)
                print(f"🚀 MPS device test: {test_tensor.device}")
            except Exception as e:
                print(f"⚠️  MPS device test failed: {e}")
        else:
            print("⚠️  Apple Silicon GPU acceleration not available")
    else:
        print("ℹ️  MPS (Metal) not supported in this PyTorch version")
    
    # Determine what device will be used
    if cuda_available:
        default_device = "CUDA GPU"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = "Apple Silicon GPU (MPS)"
    else:
        default_device = "CPU only"
    
    print(f"\n🎯 Default device for models: {default_device}")
    
    # Memory information (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 System RAM: {memory.total // (1024**3):.1f} GB (Available: {memory.available // (1024**3):.1f} GB)")
    except ImportError:
        print("💾 System RAM: Unable to detect (install psutil for memory info)")
    
    print("=" * 50)

def main():
    print("=== LOCAL MODEL TESTING SCRIPT ===")
    print("Testing locally downloaded models (offline mode)")
    
    # Enable offline mode with standardized enforcement
    enforce_offline_mode()
    
    # Show system info
    check_system_info()
    
    # Test models
    test_embedding_model()
    test_chat_model()
    
    # Show parameter control demo
    demo_custom_llm_parameters()
    
    print("\n=== TESTING COMPLETE ===")
    print("If all tests passed, your air-gapped system is ready!")
    print("You can now use these models without internet access.")
    print("\n🎛️  See the parameter demo above to learn how to control:")
    print("   • Temperature, context window, creativity levels")
    print("   • Top-p, top-k sampling for response diversity") 
    print("   • Repetition penalty for output quality")

if __name__ == "__main__":
    main()