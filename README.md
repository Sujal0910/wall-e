```markdown
# 🤖 Wall-E: Custom 124M Parameter SLM & RAG Agent

## 🫥 Project Introduction
Wall-E is a full-stack local web application powered by a custom-built 124 million parameter Small Language Model (SLM) based on the GPT-2 Transformer architecture. Pre-trained on a 4.2 billion token subset of FineWeb and instruction fine-tuned on the Stanford Alpaca dataset, Wall-E is designed to act as a focused, local task-solving assistant. 

To overcome the inherent memory limitations of small-scale parameters, this project integrates advanced Retrieval-Augmented Generation (RAG) and live internet-search capabilities, transforming the foundational model into an agentic reading-comprehension engine.

## 📝 Project Overview and Features
* **Custom Transformer Architecture:** Built entirely in PyTorch (`model.py`), replicating the GPT-2 Small configuration (`vocab_size=50304`, `n_layer=12`, `n_head=12`, `n_embd=768`).
* **Advanced Inference Engine:** Includes engineered logit scaling, a repetition penalty (`1.2`), and optimized temperature controls (`0.4`) to effectively eliminate degenerative generation loops and reduce hallucinations.
* **Retrieval-Augmented Generation (RAG):** Uses a local `ChromaDB` vector database and `Sentence-Transformers` (`all-MiniLM-L6-v2`) to inject accurate, offline document context directly into the model's prompt.
* **Live Agentic Web Search:** Integrates the DuckDuckGo Search API to fetch real-time internet data, allowing Wall-E to accurately summarize and answer questions about current events.
* **Chit-Chat Interceptor:** A lightweight routing layer that handles basic conversational inputs (greetings) outside the Alpaca instruction-tuning scope.
* **Full-Stack Web Interface:** Served locally via a Python `Flask` backend and presented through a clean, responsive `Tailwind CSS` frontend.

## ⛏️ Tech Stack
* **Core Machine Learning:** Python, PyTorch, Tiktoken
* **Vector Database & Embeddings:** ChromaDB, Sentence-Transformers
* **Web Search APIs:** DuckDuckGo-Search (implemented using the `DDGS().text()` method for live search querying).
* **Backend Web Framework:** Flask
* **Frontend UI:** HTML5, Tailwind CSS

## 📁 Project Structure
```text
wall-e/
├── app.py                      # Flask backend, RAG pipeline, Web Search, and inference loop
├── model.py                    # PyTorch GPT-2 architecture definition
├── build_db.py                 # Script to embed offline documents into ChromaDB
├── fix_model.py                # Utility script to repair unpacked/corrupted .pt files
├── gpt2_wall_e_assistant.pt    # The 124M parameter model weights (Not included in repo)
├── walle_knowledge/            # Local ChromaDB vector storage folder
└── templates/
    └── index.html              # Tailwind CSS chat interface

```

## 🧑‍💻 Getting Started: Setup and Running Instructions

### 1. Prerequisites

Ensure you have Python 3.9+ installed on your local machine.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/Sujal0910/wall-e.git](https://github.com/Sujal0910/wall-e.git)
cd wall-e
pip install torch tiktoken flask chromadb sentence-transformers duckduckgo-search

```

*(Note: The `duckduckgo-search` library provides a simple and lightweight Python wrapper for interacting with the DuckDuckGo API).*

### 3. Model Checkpoint Setup

Place your fine-tuned model checkpoint (`gpt2_wall_e_assistant.pt`) in the root directory alongside `app.py`.

**🚨 Critical Windows Unzipping Issue:**
PyTorch `.pt` files are essentially ZIP archives under the hood. Windows may automatically assign them a zipper icon and allow you to extract them. **Do not extract the file.** If you accidentally unzipped it (resulting in a folder containing `data.pkl`, `byteorder`, etc.), use the included `fix_model.py` script to seamlessly stitch it back together:

```bash
python fix_model.py

```

### 4. Build the Local RAG Knowledge Base (Optional)

To enable offline document retrieval, initialize the vector database by running:

```bash
python build_db.py

```

### 5. Start the Application

Launch the Flask server:

```bash
python app.py

```

Open your web browser and navigate to `http://127.0.0.1:5000` to interact with Wall-E.

## 🧠 Model Architecture & Hyperparameters

* **Parameters:** 124 Million
* **Configuration:** 12 Layers, 12 Attention Heads, 768 Embedding Dimension
* **Pre-Training:** 4.2 Billion tokens (FineWeb-Edu)
* **Fine-Tuning:** Instruction-tuned via Stanford Alpaca
* **Inference Tuning:**
* `Temperature`: 0.4 (Minimized to prevent hallucination in structured tasks)
* `Repetition Penalty`: 1.2 (Active logit suppression to prevent degenerative loops)
* `Top-K`: 40



## 🔭 Future Scaling Roadmap

While the 124M model serves as an excellent local agent, the following system design upgrades are planned for production scaling:

1. **Containerization:** Dockerizing the Flask API and deploying via Kubernetes for horizontal scaling.
2. **Caching:** Implementing Redis caching with TTL to bypass redundant forward passes on identical queries.
3. **Task Queues:** Offloading heavy inference workloads from the Flask main thread to a Celery/RabbitMQ background worker.
4. **Foundation SLM Pivot:** Migrating the base architecture from a custom GPT-2 to a modern Small Language Model (like Qwen2.5-0.5B or SmolLM2) to drastically improve baseline semantic logic without requiring 10B+ tokens of pre-training compute.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute to the model architecture or expand the agentic toolset.
