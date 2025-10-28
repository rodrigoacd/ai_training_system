"""
AI Configuration Module
Configures teacher and student models
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TeacherConfig:
    """Configuration for teacher AI model (Groq)"""
    api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com/openai/v1"
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout: int = 30

@dataclass
class StudentConfig:
    """Configuration for student AI model (Local TinyLlama)"""
    model_name: str = "microsoft/DialoGPT-small"  # Alternative: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device: str = "auto"  # auto, cpu, cuda
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    pad_token_id: int = 50256

# @dataclass
# class StudentConfig:
#     """Configuration for student AI model (Llama 3.x)"""
#     model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # o Meta-Llama-3.1 si tienes acceso
#     device: str = "auto"  # auto, cpu, cuda
#     max_length: int = 512
#     temperature: float = 0.7
#     do_sample: bool = True
#     pad_token_id: int = None  # Llama 3.x define su propio pad_token


from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """General training configuration"""
    batch_size: int = 4
    max_epochs: int = 10
    learning_rate: float = 0.0001
    evaluation_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    
    # Data paths
    dataset_path: str = "data/sample_qa_dataset.csv"
    test_dataset_path: str = "data/test_questions.csv"
    output_dir: str = "logs"
    model_save_path: str = "models"

@dataclass
class MemoryConfig:
    """Configuration for memory module"""
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    index_type: str = "cosine"  # or "L2"
    max_entries: int = 1000
    min_score_threshold: float = 7.0
    retrieve_k: int = 3
    storage_path: str = "memory"

@dataclass
class SystemConfig:
    """Overall system configuration"""
    teacher: TeacherConfig
    student: StudentConfig
    training: TrainingConfig
    memory: MemoryConfig
    
    # System settings
    verbose: bool = True
    save_logs: bool = True
    use_gpu: bool = True

def get_default_config() -> SystemConfig:
    """Get default system configuration"""
    return SystemConfig(
        teacher=TeacherConfig(),
        student=StudentConfig(),
        training=TrainingConfig(),
        memory= MemoryConfig()
    )

def validate_config(config: SystemConfig) -> bool:
    """Validate configuration parameters"""
    if not config.teacher.api_key:
        print("Warning: GROQ_API_KEY not found in environment variables")
        return False
    
    if config.student.max_length > 2048:
        print("Warning: Student max_length might be too high for small models")
    
    return True

# Environment variable setup helper
def setup_environment():
    """Setup environment variables and paths"""
    required_dirs = [
        "data", "logs", "models", "config"
    ]
    
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Check for .env file
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv()
    
    return True