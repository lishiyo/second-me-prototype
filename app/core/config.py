import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    # Project metadata
    PROJECT_NAME: str = "Second Me Prototype"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Digital Twin Chatbot Prototype"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # Wasabi S3 settings
    WASABI_ACCESS_KEY: str = os.getenv("WASABI_ACCESS_KEY", "")
    WASABI_SECRET_KEY: str = os.getenv("WASABI_SECRET_KEY", "")
    WASABI_BUCKET: str = os.getenv("WASABI_BUCKET", "")
    WASABI_REGION: str = os.getenv("WASABI_REGION", "us-west-1")
    WASABI_ENDPOINT: str = os.getenv("WASABI_ENDPOINT", "")
    
    # Weaviate settings
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # PostgreSQL settings
    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Modal settings (for training and inference)
    MODAL_APP_NAME: str = os.getenv("MODAL_APP_NAME", "second-me")
    
    # Redis settings (for job queue)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # File upload settings
    UPLOAD_MAX_SIZE: int = int(os.getenv("UPLOAD_MAX_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [
        "txt", "pdf", "doc", "docx", "md", "rtf",
        "csv", "xls", "xlsx", "ppt", "pptx", "html",
        "json", "xml"
    ]
    
    # For MVP, we're only using one user
    DEFAULT_USER_ID: str = "1"

    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    OVERLAP: int = int(os.getenv("OVERLAP", "50"))
    
    # Training settings
    MAX_RETRY_ATTEMPTS: int = 3
    
    def validate_settings(self) -> List[str]:
        """
        Validate that all required settings are provided.
        
        Returns:
            List of missing settings
        """
        missing = []
        required_vars = [
            "WASABI_ACCESS_KEY", 
            "WASABI_SECRET_KEY", 
            "WASABI_BUCKET", 
            "WASABI_ENDPOINT",
            "WEAVIATE_URL", 
            "WEAVIATE_API_KEY",
            "DB_HOST", 
            "DB_NAME", 
            "DB_USER", 
            "DB_PASSWORD",
            "OPENAI_API_KEY"
        ]
        
        for var in required_vars:
            if not getattr(self, var):
                missing.append(var)
                
        return missing

    @property
    def db_connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL for connection."""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


# Create a global settings instance
settings = Settings() 