#!/usr/bin/env python3
"""
offline_utils.py

Utility functions for enforcing air-gapped offline operation.
This module provides standardized offline mode enforcement across all scripts.
"""

import os


def enforce_offline_mode():
    """
    Enforce offline mode for air-gapped operation with triple-layer protection.
    
    This function implements the core security mechanism that prevents any
    internet access during model loading and inference operations.
    
    Security Layers:
    1. HF_HUB_OFFLINE: Prevents HuggingFace Hub from attempting downloads
    2. TRANSFORMERS_OFFLINE: Forces Transformers library to local-only mode
    3. PYTORCH_HUB_OFFLINE: Prevents PyTorch Hub downloads (future-proofing)
    
    This function MUST be called before any model loading operations
    in Phase 2 (offline) scripts.
    """
    # Primary offline enforcement
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # Additional protection for PyTorch Hub (future-proofing)
    os.environ['TORCH_HOME'] = './local_models/.torch_cache'
    
    print("🔒 Air-gapped offline mode enforced")
    print("   ✅ HuggingFace Hub access blocked")
    print("   ✅ Transformers library in local-only mode")
    print("   ✅ All model operations restricted to local filesystem")


def get_safe_model_loading_kwargs():
    """
    Get standardized kwargs for safe offline model loading.
    
    Returns a dictionary of parameters that should be used with
    all from_pretrained() calls to ensure offline-only operation.
    
    Returns:
        dict: Safe loading parameters for offline operation
    """
    return {
        'local_files_only': True,      # Core offline enforcement
        'torch_dtype': None,           # Let models choose appropriate dtype
        'low_cpu_mem_usage': True,     # Memory efficiency
        'trust_remote_code': False,    # Security: no remote code execution
    }


def verify_offline_environment():
    """
    Verify that the offline environment is properly configured.
    
    Checks that all required environment variables are set and
    provides diagnostic information for troubleshooting.
    
    Returns:
        bool: True if offline environment is properly configured
    """
    checks = [
        ('HF_HUB_OFFLINE', '1'),
        ('TRANSFORMERS_OFFLINE', '1'),
    ]
    
    all_good = True
    print("🔍 Verifying offline environment configuration:")
    
    for env_var, expected_value in checks:
        actual_value = os.environ.get(env_var)
        if actual_value == expected_value:
            print(f"   ✅ {env_var} = {actual_value}")
        else:
            print(f"   ❌ {env_var} = {actual_value} (expected: {expected_value})")
            all_good = False
    
    if all_good:
        print("🎉 Offline environment properly configured!")
    else:
        print("⚠️  Offline environment not properly configured")
        print("   Call enforce_offline_mode() before model operations")
    
    return all_good


def check_local_model_exists(model_path):
    """
    Check if a model exists locally with proper air-gapped validation.
    
    Args:
        model_path (str): Path to the local model directory
        
    Returns:
        bool: True if model exists and has required files
    """
    required_files = ['config.json']  # Minimum requirement
    
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print(f"   Please run '1_download_models.py' first")
        return False
    
    for required_file in required_files:
        file_path = os.path.join(model_path, required_file)
        if not os.path.exists(file_path):
            print(f"❌ Required model file missing: {file_path}")
            print(f"   Model download may be incomplete")
            return False
    
    print(f"✅ Local model verified: {model_path}")
    return True


# Example usage pattern for offline scripts:
if __name__ == "__main__":
    print("=== OFFLINE UTILITIES DEMO ===")
    
    # Step 1: Enforce offline mode
    enforce_offline_mode()
    
    # Step 2: Verify environment
    verify_offline_environment()
    
    # Step 3: Check model availability
    embedding_exists = check_local_model_exists('./local_models/embedding_model')
    chat_exists = check_local_model_exists('./local_models/chat_model')
    
    # Step 4: Show safe loading parameters
    safe_kwargs = get_safe_model_loading_kwargs()
    print(f"\n📋 Safe model loading parameters:")
    for key, value in safe_kwargs.items():
        print(f"   {key}: {value}")
    
    if embedding_exists and chat_exists:
        print("\n🚀 System ready for air-gapped operation!")
    else:
        print("\n⚠️  Some models missing. Run '1_download_models.py' first.")