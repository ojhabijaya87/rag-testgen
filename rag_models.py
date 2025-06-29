MODEL_CONFIG = {
    "Select a model": {},
    "ollama-phi (Offline)": {
        "provider": "ollama",
        "model_name": "phi3:14b-medium-4k-instruct",
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "ollama-llama3 (Offline)": {
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.3,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "ollama-mistral (Offline)": {
        "provider": "ollama",
        "model_name": "mistral:7b-instruct",
        "temperature": 0.4,
        "max_tokens": 32768,
        "top_p": 0.95
    },
    "ollama-zephyr (Offline)": {
        "provider": "ollama",
        "model_name": "zephyr:7b-beta",
        "temperature": 0.2,
        "max_tokens": 3072,
        "top_p": 0.9
    },
    "ollama-mixtral (Offline)": {
        "provider": "ollama",
        "model_name": "mixtral:8x7b-instruct",
        "temperature": 0.3,
        "max_tokens": 32768,
        "top_p": 0.9
    },
    "ollama-command-r (Offline)": {
        "provider": "ollama",
        "model_name": "command-r:35b-v0.1",
        "temperature": 0.3,
        "max_tokens": 128000,
        "top_p": 0.9
    },
    "ollama-deepseek (Offline)": {
        "provider": "ollama",
        "model_name": "deepseek-coder:33b-instruct",
        "temperature": 0.2,
        "max_tokens": 16384,
        "top_p": 0.9
    },
    "zephyr-7b-beta (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "temperature": 0.2,
        "max_new_tokens": 3072,
        "task": "text-generation"
    },
    "llama3-8b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.3,
        "max_new_tokens": 8192,
        "task": "text-generation"
    },
    "llama3-70b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct",
        "temperature": 0.3,
        "max_new_tokens": 8192,
        "task": "text-generation"
    },
    "gpt-4o (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.4,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "gpt-4-turbo (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "temperature": 0.4,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "gpt-3.5-turbo (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo-0125",
        "temperature": 0.5,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "gemini-1.5-flash (Google)": {
        "provider": "google",
        "model_name": "gemini-1.5-flash",
        "temperature": 0.5,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "gemini-1.5-pro (Google)": {
        "provider": "google",
        "model_name": "gemini-1.5-pro",
        "temperature": 0.5,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "llama3-70b (Groq)": {
        "provider": "groq",
        "model_name": "llama3-70b-8192",
        "temperature": 0.3,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "mixtral-8x7b (Groq)": {
        "provider": "groq",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.3,
        "max_tokens": 32768,
        "top_p": 0.9
    },
    "claude-3-haiku (Anthropic)": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "temperature": 0.4,
        "max_tokens": 200000,
        "top_p": 0.9
    },
    "mixtral-8x22b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "temperature": 0.3,
        "max_new_tokens": 65536,
        "task": "text-generation"
    },
    "deepseek-coder-33b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-33b-instruct",
        "temperature": 0.2,
        "max_new_tokens": 16384,
        "task": "text-generation"
    },
    "claude-3-sonnet (Anthropic)": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "temperature": 0.4,
        "max_tokens": 200000,
        "top_p": 0.9
    },
    "command-r-plus (Groq)": {
        "provider": "groq",
        "model_name": "command-r-plus",
        "temperature": 0.3,
        "max_tokens": 128000,
        "top_p": 0.9
    },
    "gemini-1.0-pro (Google)": {
        "provider": "google",
        "model_name": "gemini-1.0-pro",
        "temperature": 0.5,
        "max_tokens": 32768,
        "top_p": 0.9
    },
    "phi-3-mini-128k (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-128k-instruct",
        "temperature": 0.3,
        "max_new_tokens": 128000,
        "task": "text-generation"
    }
}

MODEL_USAGE_HINTS = {
    "Select a model": "ℹ️ Please select a model to enable test generation.",
    "ollama-phi (Offline)": "🖥️ Run: `ollama pull phi3:14b-medium-4k-instruct`",
    "ollama-llama3 (Offline)": "🖥️ Run: `ollama pull llama3:70b-instruct`",
    "ollama-mistral (Offline)": "🖥️ Run: `ollama pull mistral:7b-instruct`",
    "ollama-zephyr (Offline)": "🖥️ Run: `ollama pull zephyr:7b-beta`",
    "ollama-mixtral (Offline)": "🖥️ Run: `ollama pull mixtral:8x7b-instruct`",
    "ollama-command-r (Offline)": "🖥️ Run: `ollama pull command-r:35b-v0.1`",
    "ollama-deepseek (Offline)": "🖥️ Run: `ollama pull deepseek-coder:33b-instruct`",
    "zephyr-7b-beta (Hugging Face)": "🌐 Fast 7B model - best for most use cases",
    "llama3-8b (Hugging Face)": "🌐 Efficient 8B model - good balance",
    "llama3-70b (Hugging Face)": "🌐 Meta's flagship model - highest quality",
    "gpt-4o (OpenAI)": "🚀 OpenAI's fastest multimodal model",
    "gpt-4-turbo (OpenAI)": "🧠 OpenAI's most capable model",
    "gpt-3.5-turbo (OpenAI)": "⚡ OpenAI's fastest and most affordable model",
    "gemini-1.5-flash (Google)": "⚡ Google's fastest model - great for testing",
    "gemini-1.5-pro (Google)": "🧠 Google's most capable model",
    "llama3-70b (Groq)": "⚡ Groq - 500+ tokens/sec! Requires GROQ_API_KEY",
    "mixtral-8x7b (Groq)": "⚡ Groq - MoE model at 500+ tokens/sec",
    "claude-3-haiku (Anthropic)": "🏃 Anthropic's fastest model",
    "mixtral-8x22b (Hugging Face)": "🏆 Top open-source model - 176B params",
    "deepseek-coder-33b (Hugging Face)": "💻 Specialist for test case generation",
    "claude-3-sonnet (Anthropic)": "🔍 Best for requirement analysis & edge cases",
    "command-r-plus (Groq)": "🏢 Enterprise-grade RAG optimization",
    "gemini-1.0-pro (Google)": "📊 Most consistent output quality",
    "phi-3-mini-128k (Hugging Face)": "💰 Cost-effective with large 128K context"
}
