#!/usr/bin/env python3
"""
1_download_models.py

Phase 1 of air-gapped LLM system: Model Download (ONLINE)
Run this script ONCE when you have internet access to download models locally.
This prepares your system for completely offline operation.

Usage: python 1_download_models.py

Requirements:
- Internet connection
- ~2-3 GB free disk space
- Python packages: transformers, sentence-transformers, torch
"""

import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def check_system_requirements():
    """Check system requirements for air-gapped operation"""
    print("🔍 Checking system requirements...")
    
    # Check available disk space
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (1024**3)
    
    print(f"💾 Available disk space: {free_gb}GB")
    
    if free_gb < 3:
        print("⚠️  WARNING: Less than 3GB free space available")
        print("   Recommended: At least 3GB for model downloads")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            print("❌ Aborting download")
            return False
    else:
        print("✅ Sufficient disk space available")
    
    return True

def create_directories():
    """Create necessary directories for storing models in air-gapped environment"""
    directories = [
        "./local_models",
        "./local_models/embedding_model",
        "./local_models/chat_model"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def download_embedding_model():
    """Download SentenceTransformer model for text embeddings (all-MiniLM-L6-v2)"""
    print("\n=== Downloading Embedding Model ===")
    
    # Check if model already exists to avoid re-downloading
    if os.path.exists('./local_models/embedding_model/config.json'):
        print("⚠️  Embedding model already exists at ./local_models/embedding_model")
        print("   Skipping download to avoid overwriting existing model")
        return
    
    try:
        # Download small, efficient embedding model (~80MB)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save('./local_models/embedding_model')
        print("✅ Embedding model downloaded successfully to ./local_models/embedding_model")
        print("   Size: ~80MB, suitable for CPU inference")
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")
        print("   Check internet connection and disk space")

def download_chat_model():
    """Download conversational AI model (DialoGPT-medium) with tokenizer"""
    print("\n=== Downloading Chat Model ===")
    
    # Check if model already exists to avoid re-downloading
    if os.path.exists('./local_models/chat_model/config.json'):
        print("⚠️  Chat model already exists at ./local_models/chat_model")
        print("   Skipping download to avoid overwriting existing model")
        return
    
    try:
        model_name = 'microsoft/DialoGPT-medium'
        
        # Download tokenizer first (vocabulary and encoding rules)
        print("🔄 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained('./local_models/chat_model')
        
        # Download model weights (neural network parameters)
        print("🔄 Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained('./local_models/chat_model')
        
        print("✅ Chat model downloaded successfully to ./local_models/chat_model")
        print("   Size: ~1.5GB, optimized for conversation generation")
    except Exception as e:
        print(f"❌ Failed to download chat model: {e}")
        print("   Check internet connection and ensure ~2GB free disk space")

def verify_downloads():
    """Verify that all required model files were downloaded correctly for air-gapped operation"""
    print("\n=== Verifying Downloads ===")
    
    checks = [
        ("./local_models/embedding_model/config.json", "Embedding model config"),
        ("./local_models/embedding_model/model.safetensors", "Embedding model weights"),
        ("./local_models/chat_model/config.json", "Chat model config"),
        ("./local_models/chat_model/model.safetensors", "Chat model weights"),
        ("./local_models/chat_model/tokenizer.json", "Chat model tokenizer"),
    ]
    
    all_good = True
    for path, description in checks:
        if os.path.exists(path):
            print(f"✅ {description} found")
        else:
            print(f"❌ {description} missing at {path}")
            all_good = False
    
    if all_good:
        print("\n🎉 All models verified! Air-gapped system ready.")
        create_manifest()
    else:
        print("\n⚠️  Some files are missing. Re-run this script to complete download.")

def create_manifest():
    """Create a manifest file documenting the air-gapped system configuration"""
    manifest_content = """# Air-Gapped LLM System Manifest
# Generated by 1_download_models.py

## System Configuration
- Phase 1 (Online): Model download completed
- Phase 2 (Offline): Ready for testing with 2_test_local_models.py

## Downloaded Models
1. Embedding Model: sentence-transformers/all-MiniLM-L6-v2
   - Location: ./local_models/embedding_model/
   - Size: ~80MB
   - Purpose: Text embeddings and similarity

2. Chat Model: microsoft/DialoGPT-medium  
   - Location: ./local_models/chat_model/
   - Size: ~1.5GB
   - Purpose: Conversational AI

## Security Notes
- All models stored locally in ./local_models/
- No internet access required after Phase 1
- Models loaded with local_files_only=True
- Environment variables enforce offline mode

## Usage
1. Disconnect from internet (optional but recommended for air-gapped operation)
2. Run: python 2_test_local_models.py
3. Models will load from local filesystem only
"""
    
    try:
        with open('./local_models/MANIFEST.md', 'w') as f:
            f.write(manifest_content)
        print("📄 Created system manifest: ./local_models/MANIFEST.md")
    except Exception as e:
        print(f"⚠️  Could not create manifest: {e}")

def main():
    print("=== AIR-GAPPED LLM SYSTEM - PHASE 1 (ONLINE) ===")
    print("This script downloads models for completely offline operation")
    print("Requirements: Internet connection, ~3GB disk space\n")
    
    # Check if all models already exist (quick exit for already-prepared systems)
    embedding_exists = os.path.exists('./local_models/embedding_model/config.json')
    chat_exists = os.path.exists('./local_models/chat_model/config.json')
    
    if embedding_exists and chat_exists:
        print("⚠️  All models already exist!")
        print("   📁 Embedding model: ./local_models/embedding_model/")
        print("   📁 Chat model: ./local_models/chat_model/")
        print("\n🚀 System ready for air-gapped operation!")
        print("   Next: Run 'python 2_test_local_models.py' to test offline functionality")
        print("   Or delete model directories if you want to re-download")
        return
    
    print("🌐 Starting model download process...")
    
    # Check system requirements before proceeding
    if not check_system_requirements():
        return
    
    # Create directory structure
    create_directories()
    
    # Download models (will skip if individual models exist)
    download_embedding_model()
    download_chat_model()
    
    # Verify all downloads completed successfully
    verify_downloads()
    
    print("\n=== PHASE 1 COMPLETE ===")
    print("🔒 Models are now saved locally for air-gapped operation")
    print("📴 You can now disconnect from the internet")
    print("🚀 Next: Run 'python 2_test_local_models.py' to test offline functionality")

if __name__ == "__main__":
    main()