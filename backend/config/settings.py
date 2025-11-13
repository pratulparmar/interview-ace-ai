"""
Configuration settings module

What this File does :

- Loads API keys from the .env file
- Stores all app settings in one place
- validates that the required setting exist

CONTROL PANEL of the APP

"""

# ========== IMPORTS ==========
# These are like "ingredients" - we're importing tools we need

from pydantic_settings import BaseSettings, SettingsConfigDict  # Helps Manage Settings
from pydantic import Field                  # Helps Validate Settings
from typing import Optional                 # Means - This setting is optional
from pathlib import Path                    # Helps Work with the file path
import os                                   # Operating System tools


# ========== FIND PROJECT ROOT ==========
# __file__ = this current file's location
# .parent = go up one folder level
# .parent.parent = go up two folder levels (to project root)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ========== SETTINGS CLASS ==========

class Settings(BaseSettings):
    """
    All Application Settings

    HOW IT WORKS:
    1. Looks for .env file in project root
    2. Reads variables from .env
    3. Stores them as settings we can access

    ex.:
    If .env has : OPENAI_API_KEY = sk-****
    we can access it as : settings.openai_api_key
    """

    # ===== OpenAI Settings =====

    openai_api_key : str = Field(
        ...,  # The "....," means REQUIRED - app won't start without it
        description = "Your OpenAI API key"
    )

    openai_model : str = Field(
        default="gpt-4-turbo-preview",  # Default value if not in .env
        description="Which GPT model to use"
    )
    
    openai_temperature: float = Field(
        default=0.7,
        description="0.0 = very focused, 2.0 = very creative"
    )

    
    # ===== LangSmith Settings (for monitoring) =====

    langchain_tracing_v2: bool = Field(
        default = True,
        description = "Turn LangSmith monitoring on/off"
    )

    langchain_api_key: Optional[str] = Field(
        default=None,  # Optional means it can be None (empty)
        description="LangSmith API key (optional but recommended)"
    )

    langchain_project: str = Field(
        default="interview-ace-ai",
        description="Project name in LangSmith"
    )


    # ===== Vector Database Settings =====

    vector_db_type: str = Field (
        default = "chroma",
        description = "We'll use ChromaDB (runs locally, no API key needed)"
    )

    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma_db",
        description="Where ChromaDB saves data on your computer"
    )


    # ===== Regular Database Settings =====

    database_url: str = Field(
        default="sqlite:///./data/interview_ace.db",
        description="SQLite = simple database, stored as a file"
    )

    # ===== API Settings =====

    backend_port: int = Field(
        default=8000,
        description="Your API will run on localhost:8000"
    )

    debug_mode: bool = Field(
        default=True,
        description="Extra logging to help you learn"
    )


    # ===== RAG Settings =====

    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Model that converts text to numbers (vectors)"
    )
    
    top_k_results: int = Field(
        default=5,
        description="How many similar documents to retrieve"
    )


    # ===== Pydantic Config =====

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # NEW: Ignores extra/unknown fields
    )


# ========== SINGLETON PATTERN ==========
# WHAT: Only create Settings object ONCE, then reuse it everywhere
# WHY: Loading .env is slow - do it once!   

_settings = None   # Global variable to store settings

def get_settings() -> Settings:
    """
    Get Settings (Creates them if first time)

    USAGE:
        from backend.config.settings import get_settings
        settings = get_settings()
        print(settings.openai_api_key)

    """

    global _settings

    if _settings is None:
        print("Lading settings from .env file....")
        _settings = Settings()

        # Create data folders if they don't exist
        _settings.chroma_persist_directory.mkdir(parents= True, exist_ok= True)
        (PROJECT_ROOT / "data" / "interview_questions").mkdir(parents= True, exist_ok= True)
        (PROJECT_ROOT / "logs").mkdir(parents= True, exist_ok= True)

        print("Settings loaded successfully")
    return _settings


def setup_langsmith():
    """
    Enabling Langsmith Tracing

    What it does:
    - Tracks every LLM call we make
    - Shows us prompts, response, timing, cost
    - super helpful for debugging !

    USAGE:
        setup_langsmith()  # Call this once at the app startup
    """ 

    settings = get_settings()

    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        # Set environment variable so LangChain knows to use LangSmith
        os.environ["LANGCHAIN_TRACING_V2"]= "true"
        os.environ["LANGCHAIN_API_KEY"] = str(settings.langchain_api_key)
        os.environ["LANGCHAIN_PROJECT"] = str(settings.langchain_project)

        print("LangSmith monitoring enabled!")
        print(f"View traces at: https://smith.langchain.com/")
        
    else:
        print("LangSmith disabled (no API key found)")
        print("Get free key at : https://smith.langchain.com/")


# ========== TEST CODE ==========
# This runs ONLY if you run this file directly

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Configuration")
    print("\n" + "="*50)


    try:
        # Try to load settings
        settings = get_settings()

        print("📋 Configuration Summary:")
        print(f"   App Name: InterviewAce AI")
        print(f"   OpenAI Model: {settings.openai_model}")
        print(f"   Temperature: {settings.openai_temperature}")
        print(f"   Vector DB: {settings.vector_db_type}")
        print(f"   Backend Port: {settings.backend_port}")
        print(f"   Debug Mode: {settings.debug_mode}")
        print(f"   Data Directory: {settings.chroma_persist_directory}")
        
        # Check if API key is set
        if settings.openai_api_key == "your-key-will-go-here":
            print("\n WARNING: Using placeholder API key!")
            print("   Replace it in .env with your real key from:")
            print("   https://platform.openai.com/api-keys")
        else:
            print(f"\n OpenAI API Key: {settings.openai_api_key[:20]}...")
        
        # Setup LangSmith
        print("\n" + "="*50)
        try:
            setup_langsmith()
        except Exception as e:
            print(f"LangSmith setup error: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

        print("="*50)
        
        print("\n Configuration test PASSED!")
        print("\n Next steps:")
        print("   1. Get your OpenAI API key")
        print("   2. Add it to .env file")
        print("   3. Run this test again")
        
    except Exception as e:
        print(f"\n Configuration test FAILED!")
        print(f"   Error: {e}")
        print("\n Troubleshooting:")
        print("   1. Make sure .env file exists in project root")
        print("   2. Make sure it has OPENAI_API_KEY=your_key")
        print("   3. Make sure you ran: pip install -r backend/requirements.txt")
        print("   4. Make sure venv is activated (you see '(venv)' in terminal)")
          