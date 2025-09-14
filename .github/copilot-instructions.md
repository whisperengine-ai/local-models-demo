# Copilot Instructions for Air-Gapped LLM System

## System Architecture

This is a **two-phase air-gapped LLM system** designed for secure, offline operation:

1. **Phase 1 (Online)**: `1_download_models.py` downloads models to `./local_models/` with system validation
2. **Phase 2 (Offline)**: `2_test_local_models.py` tests models with enforced offline mode

### Critical Environment Variables
Always use these offline mode flags in Phase 2:
```python
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

## Model Organization Pattern

Models are stored in `./local_models/` with specific directory structure:
- `embedding_model/` - SentenceTransformer (all-MiniLM-L6-v2, ~80MB)
- `chat_model/` - DialoGPT-medium for conversations (~1.5GB)
- `MANIFEST.md` - System configuration documentation

Each model directory contains standard HuggingFace files: `config.json`, tokenizer files, and model weights (`.safetensors`).

## Loading Patterns

**Always use `local_files_only=True`** when loading models:
```python
# For transformers
tokenizer = AutoTokenizer.from_pretrained('./local_models/chat_model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('./local_models/chat_model', local_files_only=True)

# For sentence-transformers
model = SentenceTransformer('./local_models/embedding_model', device='cpu')
```

## Critical Development Workflows

### System Requirements Check
Phase 1 validates disk space (requires 3GB+) and system readiness before downloading.

### Testing Offline Functionality
Run `python 2_test_local_models.py` to verify air-gapped operation. This script:
- Sets offline environment variables
- Tests both model types sequentially
- Validates embeddings with similarity calculations
- Generates chat responses with proper tokenization

### Model Download Preparation
Run `python 1_download_models.py` once with internet access. Features:
- System requirements validation
- Skip logic for existing models
- Comprehensive verification with multiple file checks
- Automatic manifest generation

## Error Handling Patterns

Models fail gracefully when not found:
```python
if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("   Please run '1_download_models.py' first")
    return
```

Disk space validation prevents incomplete downloads:
```python
free_gb = shutil.disk_usage(".")[2] // (1024**3)
if free_gb < 3:
    print("⚠️  WARNING: Less than 3GB free space available")
```

## Memory Management
- Use `torch.float16` for large models
- Set `device_map="cpu"` for CPU-only operation
- Enable `low_cpu_mem_usage=True` for memory-constrained environments
- Handle missing pad tokens: `tokenizer.pad_token = tokenizer.eos_token`

## Security Constraints
- No network access after Phase 1
- All computation happens locally
- Environment variables enforce offline mode
- MANIFEST.md documents system configuration
- Suitable for classified/sensitive environments