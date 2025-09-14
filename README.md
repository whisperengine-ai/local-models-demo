# Air-Gapped LLM System Setup

This repository contains scripts to set up and run Large Language Models (LLMs) in a completely air-gapped environment with no internet access re### **Standardized Offline Enforcement**

Use the included `offline_utils.py` for consistent protection across all scripts:

```python
from offline_utils import enforce_offline_mode, get_safe_model_loading_kwargs, check_local_model_exists

# 1. Enforce offline mode (call this first)
enforce_offline_mode()

# 2. Check model exists locally
if not check_local_model_exists('./local_models/chat_model'):
    exit("Model not found - run 1_download_models.py first")

# 3. Load with standardized safe parameters
safe_kwargs = get_safe_model_loading_kwargs()
tokenizer = AutoTokenizer.from_pretrained('./local_models/chat_model', **safe_kwargs)
model = AutoModelForCausalLM.from_pretrained('./local_models/chat_model', **safe_kwargs)
```

**Safe loading parameters automatically include:**
- `local_files_only=True` - Core offline enforcement
- `low_cpu_mem_usage=True` - Memory efficiency  
- `trust_remote_code=False` - Security (no remote code execution)

### Text Embeddings
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./local_models/embedding_model')
embeddings = model.encode(["Your text here"])
```

### Chat Completion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from offline_utils import enforce_offline_mode, get_safe_model_loading_kwargs

# Enforce offline mode first
enforce_offline_mode()

# Load with safe parameters
safe_kwargs = get_safe_model_loading_kwargs()
tokenizer = AutoTokenizer.from_pretrained('./local_models/chat_model', **safe_kwargs)
model = AutoModelForCausalLM.from_pretrained('./local_models/chat_model', **safe_kwargs) initial setup.

## Quick Start

### Step 1: Setup (With Internet)
```bash
# Install dependencies
pip install -r requirements.txt

# Download models for offline use
python 1_download_models.py
```

### Step 2: Test Offline
```bash
# Disconnect from internet, then:
python 2_test_local_models.py

# Try interactive LLM with parameter control:
python example_llm_usage.py
```

## Files Description

- **`1_download_models.py`** - Downloads models to local filesystem (run once with internet)
- **`2_test_local_models.py`** - Tests local models in offline mode with parameter demos
- **`example_llm_usage.py`** - Interactive LLM usage with parameter control
- **`LLM_PARAMETER_GUIDE.md`** - Complete parameter reference guide
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This setup guide

## System Requirements

- **Storage**: ~3 GB for models (80MB embedding + 1.5GB chat model)
- **RAM**: 4 GB minimum, 8 GB recommended for optimal performance  
- **Python**: **3.9 or higher** (tested with Python 3.9-3.13)

### Platform Compatibility

✅ **Cross-Platform Support**
- **Linux**: Full support with CUDA GPU acceleration (NVIDIA GPUs)
- **Windows**: Full support with CUDA GPU acceleration (NVIDIA GPUs) 
- **macOS**: Full support with Apple Silicon GPU acceleration (MPS)

**GPU Acceleration Automatically Detected:**
- **NVIDIA GPUs**: Uses CUDA (Linux/Windows)
- **Apple Silicon**: Uses Metal Performance Shaders (macOS M1/M2/M3/M4)
- **CPU Fallback**: Works on any system without GPU

## Model Selection & Rationale

### 🔤 **Embedding Model: all-MiniLM-L6-v2**
- **Size**: ~80MB (very lightweight)
- **Purpose**: Text embeddings and semantic similarity
- **Why this model?**
  - ✅ **Compact**: Perfect for air-gapped environments with storage constraints
  - ✅ **Fast**: Optimized for CPU inference, works well without GPU
  - ✅ **Versatile**: 384-dimensional embeddings work for most similarity tasks
  - ✅ **Battle-tested**: Popular choice with broad language coverage
  - ✅ **Local-friendly**: No special tokenization requirements

### 💬 **Chat Model: DialoGPT-medium**
- **Size**: ~1.5GB (balanced size/performance)
- **Purpose**: Conversational AI and text generation
- **Why this model?**
  - ✅ **Air-gap optimized**: Designed for offline deployment
  - ✅ **Conversation-focused**: Specifically trained for dialogue (not just completion)
  - ✅ **Memory efficient**: Medium size balances quality vs. resource usage
  - ✅ **No external dependencies**: Self-contained without API requirements
  - ✅ **Local tokenizer**: Includes complete vocabulary for offline operation
  - ✅ **Proven stability**: Mature model with extensive real-world usage

### 🎯 **Alternative Models Considered (& Why Not Chosen)**

**Larger Models (GPT-3.5/4 scale):**
- ❌ 10-50GB+ sizes impractical for air-gapped distribution
- ❌ Require significant GPU memory (16GB+ VRAM)
- ❌ Slower inference on CPU-only systems

**Smaller Models (GPT-2):**
- ❌ Lower conversation quality
- ❌ Limited context understanding
- ❌ Less coherent multi-turn dialogues

**API-based Models:**
- ❌ Require internet connection (defeats air-gap purpose)
- ❌ Data privacy concerns for sensitive environments
- ❌ No offline operation capability

### 📊 **Size vs. Performance Trade-offs**

| Model Type | Size | CPU Performance | GPU Performance | Use Case |
|------------|------|-----------------|-----------------|----------|
| Embedding | 80MB | Excellent | Excellent | Similarity, search, clustering |
| Chat Medium | 1.5GB | Good | Very Good | Conversations, Q&A, assistance |

**Total footprint: ~1.6GB** - Fits comfortably on most systems while providing solid AI capabilities for air-gapped environments.

## Air-Gapped Operation & Security

After downloading models, the system operates completely offline with **multi-layer protection**:

### 🔒 **Offline Mode Enforcement**

**Triple-Layer Protection:**
1. **Environment Variables** (Primary)
   ```python
   os.environ['HF_HUB_OFFLINE'] = '1'        # Block HuggingFace Hub access
   os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force local-only operation
   ```

2. **Local Files Only** (Secondary)
   ```python
   # Every model load enforces local-only access
   tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
   model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
   ```

3. **Local Path References** (Tertiary)
   ```python
   model_path = './local_models/chat_model'  # Filesystem paths, not URLs
   ```

### 🛡️ **Security Guarantees**

✅ **No internet connection required**  
✅ **No external API calls possible**  
✅ **All models loaded from local filesystem**  
✅ **Immediate failure if files missing** (no silent fallbacks)  
✅ **Suitable for classified/sensitive environments**  

### 🔍 **Verify Offline Mode**

```bash
# Test 1: Disconnect internet, then run
python 2_test_local_models.py

# Test 2: Check environment variables are set
python -c "import os; print('Offline mode:', os.environ.get('HF_HUB_OFFLINE'))"

# Test 3: Monitor network activity (should show none)
sudo lsof -i -P | grep python
```

**Error Example** (if internet attempted):
```
OSError: We are in offline mode, but we cannot find the model files.
```

## Directory Structure

After running the download script:

```
project/
├── 1_download_models.py
├── 2_test_local_models.py
├── requirements.txt
├── README.md
└── local_models/
    ├── MANIFEST.md         # System configuration documentation
    ├── embedding_model/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── ...
    └── chat_model/
        ├── config.json  
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

## Usage Examples

### Generate Embeddings
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./local_models/embedding_model')
embeddings = model.encode(["Your text here"])
```

### Chat Completion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('./local_models/chat_model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('./local_models/chat_model', local_files_only=True)

# Generate with context window control
def generate_response(prompt, max_new_tokens=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,  # Controls output length
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
```

### Context Window Control
```python
# Control how much the model generates
response = generate_response("Explain AI", max_new_tokens=30)  # Short response
response = generate_response("Tell a story", max_new_tokens=100)  # Longer response

# Manage conversation context
conversation_history = ""
for user_input in ["Hello", "How are you?", "What's AI?"]:
    full_context = conversation_history + f"User: {user_input}"
    response = generate_response(full_context, max_new_tokens=40)
    conversation_history += f"User: {user_input} Bot: {response} "
```

## Troubleshooting

**Problem**: "Model not found" error
- **Solution**: Run `1_download_models.py` first with internet connection

**Problem**: Model trying to access internet
- **Solution**: Ensure `HF_HUB_OFFLINE=1` environment variable is set

**Problem**: Out of memory
- **Solution**: Use smaller models or enable CPU offloading

## Security Notes

- ✅ No network access after initial download
- ✅ All computation happens locally
- ✅ No data sent to external servers
- ✅ Suitable for classified/sensitive environments

## Advanced Options

### LLM Parameter Control
Control creativity, response length, and behavior:
```bash
# See parameter demonstrations
python 2_test_local_models.py

# Interactive parameter testing
python example_llm_usage.py

# Read complete parameter guide
cat LLM_PARAMETER_GUIDE.md
```

**Key Parameters:**
- **`max_new_tokens`**: Controls response length (10-500 tokens)
- **`temperature`**: Controls creativity (0.1=focused, 1.5=creative)
- **Context window**: DialoGPT-medium supports ~1024 tokens total
- **Conversation memory**: Manage chat history for context

### Context Window Management
```python
# Basic length control
model.generate(inputs, max_new_tokens=50)  # Short responses
model.generate(inputs, max_new_tokens=200) # Longer responses

# Conversation context
# Keep recent conversation history within token limits
if token_count > 800:  # Leave room for new response
    truncate_old_messages()
```

### CPU-Only Operation
```bash
export CUDA_VISIBLE_DEVICES=""
python 2_test_local_models.py
```

### Memory-Efficient Loading

Set environment variables for lower memory usage:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python 2_test_local_models.py
```

## Cross-Platform Notes

### Linux Installation
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# RHEL/CentOS/Fedora  
sudo dnf install python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Windows Installation
```cmd
REM Install Python 3.9+ from python.org
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS Installation  
```bash
# Install Python via Homebrew (recommended)
brew install python@3.11
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**File Path Compatibility:**
- Uses `os.path` and `pathlib` for cross-platform file handling
- All paths work identically on Windows (`\`), Linux/macOS (`/`)
- Models stored in `./local_models/` on all platforms

**GPU Detection Logic:**
1. Checks for CUDA (NVIDIA) - Linux/Windows primarily
2. Checks for MPS (Apple Silicon) - macOS M1/M2/M3/M4  
3. Falls back to CPU - Works everywhere

The system automatically adapts to your platform and available hardware!
Models can be loaded with reduced memory usage:
- Use `torch.float16` instead of `torch.float32`
- Enable `low_cpu_mem_usage=True`
- Consider quantized models (GGUF format)

## Support

For issues or questions:
1. Check that all requirements are installed
2. Verify models were downloaded completely
3. Ensure offline mode environment variables are set
4. Check available disk space and memory