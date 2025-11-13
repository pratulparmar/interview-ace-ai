# 🎯 InterviewAce AI

**AI-Powered Technical Interview Assistant with RAG, Multi-Agent System, and LangChain**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready GenAI application that conducts technical interviews, evaluates answers using AI agents, and provides comprehensive feedback with detailed reports.

---

## 🌟 Features

### Core Capabilities
- **🔍 RAG-Powered Question Retrieval** - Semantic search through 40+ curated interview questions
- **🤖 AI Question Generation** - GPT-4 generates custom questions based on job descriptions
- **🎤 Interactive Interviews** - Conduct full technical interviews via CLI or Web UI
- **📊 AI Evaluation** - Multi-criteria scoring with detailed feedback
- **📈 Comprehensive Reports** - Overall scores, category breakdown, and actionable insights

### Technical Highlights
- **Multi-Agent System** - Specialized agents for generation, interviewing, and evaluation
- **Vector Database** - FAISS for fast semantic search
- **LangChain Integration** - Production-ready LLM orchestration
- **Prompt Engineering** - Structured outputs with rubric-based evaluation
- **Hybrid Approach** - RAG retrieval + AI generation for optimal results

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    InterviewAce AI System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Job Description │──→│ RAG Retriever │──→│  Questions   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                     │                               │
│         │             ┌───────────────┐                       │
│         └────────────→│   Generator   │                       │
│                       │     Agent     │                       │
│                       └───────────────┘                       │
│                               │                               │
│                               ↓                               │
│                       ┌───────────────┐                       │
│                       │  Interviewer  │                       │
│                       │     Agent     │                       │
│                       └───────────────┘                       │
│                               │                               │
│                               ↓                               │
│                       ┌───────────────┐                       │
│                       │   Evaluator   │                       │
│                       │     Agent     │                       │
│                       └───────────────┘                       │
│                               │                               │
│                               ↓                               │
│                       ┌───────────────┐                       │
│                       │  Final Report │                       │
│                       └───────────────┘                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **RAG Retriever** | FAISS + OpenAI Embeddings | Semantic search through question bank |
| **Question Generator** | GPT-4 + LangChain | Generate custom interview questions |
| **Interviewer Agent** | GPT-4 + LangChain | Conduct natural conversations |
| **Evaluator Agent** | GPT-4 + Structured Outputs | Score answers with detailed feedback |
| **Vector Store** | FAISS | Store and retrieve 3072-dim embeddings |
| **Orchestrator** | Python | Coordinate all components |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/interview-ace-ai.git
   cd interview-ace-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_api_key_here
   ```

5. **Initialize question bank**
   ```bash
   python backend/scripts/question_bank_initializer.py
   ```
   This creates a vector database with 40+ curated questions (takes ~1 minute).

---

## 💻 Usage

### Option 1: Interactive CLI

Run an interview in your terminal:

```bash
python cli_interview.py
```

**Features:**
- Interactive Q&A
- Real-time answer submission
- Immediate evaluation
- Detailed feedback

**Demo:**
```
🎯 INTERVIEW ACE AI - Interactive Technical Interview

Enter your name: John Doe
Job Title: Senior Python Engineer

[Paste job description]
[Answer 5 questions]
[Receive comprehensive evaluation]
```

### Option 2: Streamlit Web UI

Launch the web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Beautiful, intuitive interface
- Progress tracking
- Question-by-question navigation
- Visual score breakdown
- Download reports (JSON/TXT)

### Option 3: Programmatic API

Use the orchestrator directly in your code:

```python
from backend.orchestrator.interview_orchestrator import InterviewOrchestrator

# Initialize
orchestrator = InterviewOrchestrator()

# Run complete interview
session = orchestrator.run_complete_interview(
    candidate_name="Alice Smith",
    job_description="Senior Python Engineer with ML experience...",
    job_title="Senior Python Engineer",
    num_questions=5,
    simulate=False,  # Set to True for testing
    use_rag=True
)

# Get report
print(session.detailed_feedback)

# Save session
orchestrator.save_session(session)
```

---

## 📊 Sample Output

```
================================================================================
INTERVIEW REPORT
================================================================================

CANDIDATE INFORMATION
---------------------
Name: John Doe
Position: Senior Python Engineer
Interview Date: 2025-11-08T...

OVERALL PERFORMANCE
-------------------
Overall Score: 7.5/10
Recommendation: Hire - Solid performance with minor gaps

CATEGORY BREAKDOWN
------------------
Python......................... 8.0/10
LLMs........................... 7.5/10
RAG............................ 7.0/10
LangChain...................... 7.5/10

QUESTION-BY-QUESTION ANALYSIS
------------------------------

Question 1: What is RAG and why is it useful?
Score: 7.5/10
  Technical Accuracy: 8/10
  Completeness: 7/10
  Communication: 8/10
  Depth: 7/10

  Strengths:
    ✓ Clear explanation of RAG fundamentals
    ✓ Mentioned key use cases

  Areas for Improvement:
    → Could discuss implementation details
    → Add more real-world examples

[... more questions ...]

FINAL RECOMMENDATION
--------------------
Hire - Solid performance with minor gaps
================================================================================
```

---

## 🗂️ Project Structure

```
interview-ace-ai/
├── backend/
│   ├── agents/
│   │   ├── question_generator.py    # GPT-4 question generation
│   │   ├── interviewer_agent.py     # Conversation management
│   │   └── evaluator_agent.py       # Answer scoring
│   ├── rag/
│   │   ├── embeddings.py            # OpenAI embeddings wrapper
│   │   ├── vector_store.py          # FAISS vector database
│   │   └── rag_retriever.py         # Semantic search logic
│   ├── orchestrator/
│   │   └── interview_orchestrator.py # Main pipeline coordinator
│   ├── config/
│   │   └── settings.py              # Configuration management
│   ├── scripts/
│   │   └── question_bank_initializer.py # Populate vector DB
│   └── data/
│       ├── vector_store/            # FAISS index + metadata
│       └── interviews/              # Saved interview sessions
├── cli_interview.py                 # Interactive CLI application
├── app.py                           # Streamlit web UI
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LangSmith Tracing (for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=interview-ace-ai
```

### Customization

**Add more questions:**
```python
# Edit backend/scripts/question_bank_initializer.py
# Add your questions to the appropriate category function
```

**Adjust evaluation criteria:**
```python
# Edit backend/agents/evaluator_agent.py
# Modify the rubric in the prompt
```

**Change models:**
```python
# Edit backend/config/settings.py
# Modify model settings (e.g., use gpt-3.5-turbo for cost savings)
```

---

## 🧪 Testing

### Test Individual Components

**Test embeddings:**
```bash
python backend/rag/embeddings.py
```

**Test vector store:**
```bash
python backend/rag/vector_store.py
```

**Test RAG retriever:**
```bash
python backend/rag/rag_retriever.py
```

**Test complete pipeline:**
```bash
python backend/orchestrator/interview_orchestrator.py
```

### Run Full System Test

```bash
# With simulated answers
python backend/orchestrator/interview_orchestrator.py

# Interactive test
python cli_interview.py
```

---

## 💰 Cost Estimation

| Operation | Cost per 1000 tokens | Typical Interview Cost |
|-----------|---------------------|------------------------|
| Embeddings (3072-dim) | $0.00013 | ~$0.05 |
| GPT-4 Question Gen | $0.01 input / $0.03 output | ~$0.15 |
| GPT-4 Evaluation | $0.01 input / $0.03 output | ~$0.50 |
| **Total per 5-question interview** | | **~$0.70** |

**Tips to reduce costs:**
- Use RAG retrieval (avoids generation cost)
- Switch to GPT-3.5-turbo for evaluation (~10x cheaper)
- Batch process multiple interviews
- Cache embeddings (already implemented)

---

## 🎓 What This Project Demonstrates

### GenAI Engineering Skills
✅ **RAG Implementation** - Vector DB, semantic search, retrieval strategies  
✅ **Multi-Agent Systems** - Specialized agents with distinct roles  
✅ **LLM Orchestration** - LangChain chains, prompts, and structured outputs  
✅ **Prompt Engineering** - Few-shot learning, CoT, structured JSON  
✅ **Vector Databases** - FAISS integration and optimization  
✅ **Production Architecture** - Modular, scalable, testable design  

### Software Engineering Skills
✅ **Clean Code** - Type hints, docstrings, separation of concerns  
✅ **System Design** - Component-based architecture  
✅ **Error Handling** - Graceful failures and user feedback  
✅ **Testing** - Unit tests for each component  
✅ **Documentation** - Comprehensive README and code comments  
✅ **UX Design** - Both CLI and Web interfaces  

---

## 🛠️ Technologies Used

- **LLMs:** OpenAI GPT-4, GPT-3.5-turbo
- **Embeddings:** OpenAI text-embedding-3-large (3072 dimensions)
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Framework:** LangChain, LangGraph
- **Web UI:** Streamlit
- **Language:** Python 3.9+
- **Monitoring:** LangSmith (optional)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Enhancement
- [ ] Add more question categories (System Design, Behavioral, etc.)
- [ ] Implement voice interview mode (speech-to-text)
- [ ] Add multi-language support
- [ ] Create analytics dashboard
- [ ] Implement user authentication
- [ ] Add interview recording/replay
- [ ] Build mobile app version

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- OpenAI for GPT-4 and embeddings
- LangChain team for the excellent framework
- Facebook Research for FAISS
- Streamlit for the web framework

---

## 📧 Contact

**Your Name**  
- GitHub: [@pratulparmar](https://github.com/pratulparmar)
- LinkedIn: [Pratul Parmar](https://www.linkedin.com/in/pratul-parmar-a5002417a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BOW%2B1F9BwTm6kxUwnRt92yg%3D%3D)
- Email: pratulparmar8@gmail.com

---

## ⭐ Star this repo

If you find this project helpful, please give it a star! It helps others discover this work.

---

**Built with ❤️ as a portfolio project demonstrating GenAI engineering skills**