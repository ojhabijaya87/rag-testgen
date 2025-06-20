# 🚇 TfL Journey Planner Test Generator

Generate BDD-style test cases (Gherkin format) using AI from user stories and live TfL website content.

Supports:
- 💻 Local models (Phi-2, Falcon)
- 🧠 Ollama-based LLMs (Zephyr, Phi, LLaMA3, Mistral)
- ☁️ Hugging Face-hosted models (Zephyr-7b-beta)

---

## 🛠 Features

- 📄 Loads and scrapes content from TfL journey planning pages
- 🔍 Retrieves context via FAISS vector search
- ✍️ Auto-generates test cases using LLMs (locally or remotely)
- ⚙️ Choose between multiple models: local, Ollama, or Hugging Face
- 🧪 Outputs Gherkin-based test cases for easy BDD automation

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourname/tfl-testgen.git
cd tfl-testgen
