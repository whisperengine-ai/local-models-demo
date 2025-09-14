# LLM Parameter Control Reference Guide

## 🎛️ Key Parameters

### **Temperature** (0.1 - 2.0)
Controls randomness and creativity:
- **0.1-0.3**: Very focused, deterministic (good for facts, code)
- **0.5-0.7**: Moderate creativity (professional writing)  
- **0.8-1.0**: Balanced (natural conversation)
- **1.2-2.0**: Very creative (stories, brainstorming)

### **Max New Tokens** (10 - 2048)
Controls response length:
- **10-30**: Short, concise answers
- **50-100**: Medium responses  
- **100-500**: Long, detailed responses
- **500+**: Very long content (essays, stories)

### **Top-p** (0.1 - 1.0) 
Nucleus sampling - probability mass:
- **0.7**: More focused vocabulary
- **0.9**: Balanced diversity
- **0.95**: Wide vocabulary range

### **Top-k** (1 - 100)
Number of candidate tokens:
- **20**: Very focused
- **50**: Balanced
- **100**: Very diverse

### **Repetition Penalty** (1.0 - 2.0)
Prevents repetitive output:
- **1.0**: No penalty (may repeat)
- **1.1-1.3**: Good balance
- **1.5+**: Strong anti-repetition

## 🎯 Recommended Presets

### 📚 **Factual Q&A**
```python
{
    'temperature': 0.3,
    'max_new_tokens': 50,
    'top_p': 0.7,
    'top_k': 20,
    'repetition_penalty': 1.2
}
```

### 🎨 **Creative Writing**
```python
{
    'temperature': 1.2,
    'max_new_tokens': 100,
    'top_p': 0.95,
    'top_k': 100,
    'repetition_penalty': 1.1
}
```

### 💼 **Professional Communication**
```python
{
    'temperature': 0.5,
    'max_new_tokens': 75,
    'top_p': 0.8,
    'top_k': 40,
    'repetition_penalty': 1.3
}
```

### 🗨️ **Casual Chat**
```python
{
    'temperature': 0.8,
    'max_new_tokens': 50,
    'top_p': 0.9,
    'top_k': 50,
    'repetition_penalty': 1.1
}
```

## 📝 Quick Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (run once)
tokenizer = AutoTokenizer.from_pretrained('./local_models/chat_model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('./local_models/chat_model', local_files_only=True)

# Generate function
def generate_response(prompt, **params):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + params.get('max_new_tokens', 50),
            temperature=params.get('temperature', 0.8),
            top_p=params.get('top_p', 0.9),
            top_k=params.get('top_k', 50),
            do_sample=True,
            repetition_penalty=params.get('repetition_penalty', 1.1),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

# Use it
response = generate_response("Explain AI", temperature=0.3, max_new_tokens=40)
```

## 🔧 Troubleshooting

### Model too repetitive?
→ Increase `repetition_penalty` to 1.3-1.5

### Responses too random?
→ Lower `temperature` to 0.3-0.5

### Responses too short?
→ Increase `max_new_tokens`

### Want more creativity?
→ Increase `temperature` and `top_p`

### Want more focus?
→ Decrease `temperature` and `top_k`

## 🚀 Performance Tips

- Start with balanced presets, then adjust
- Test different combinations for your use case
- Higher temperature = more GPU memory usage
- Longer max_new_tokens = slower generation
- Use temperature=0 for deterministic output

## 📱 Interactive Testing

Run `python example_llm_usage.py` and choose option 2 for interactive parameter testing!