"""
AI Training System - Main Application
Entry point for the AI training and evaluation system
"""
import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.ai_config import get_default_config, validate_config, setup_environment
from src.orchestrator import TrainingOrchestrator
from src.evaluator import ModelEvaluator
from src.student import StudentAI
from src.utils.data_manager import DataManager

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    AI TRAINING SYSTEM                        ║
    ║              Teacher-Student AI Learning Platform            ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Components:
    • Teacher AI: Groq Llama-3-70B (Evaluation & Guidance)
    • Student AI: Small Local Model (Learning & Responses)
    • Training Data: Q&A Dataset with Progressive Learning
    
    """
    print(banner)

def show_main_menu():
    """Display main menu options"""
    menu = """
    ═══════════════ MAIN MENU ═══════════════
    
    1. 🎯 Run Training Session
    2. 📊 Run Evaluation Only  
    3. 💬 Interactive Chat with Student
    4. 🎮 Interactive Training Mode
    5. 📈 Generate Performance Report
    6. ⚙️  System Configuration
    7. 🔍 Test Dataset Manager
    8. ❓ Help & Documentation
    9. 🚪 Exit
    
    ════════════════════════════════════════════
    """
    print(menu)

def run_training_session(orchestrator):
    """Run a training session"""
    print("\ TRAINING SESSION")
    print("-" * 40)
    
    try:
        num_questions = input("Number of training questions (default 5): ").strip()
        num_questions = int(num_questions) if num_questions else 5
        
        print(f"\nStarting training with {num_questions} questions...")
        result = orchestrator.run_training_cycle(num_questions)
        
        if result["success"]:
            stats = result["statistics"]
            print(f"\nTraining completed successfully!")
            print(f"   Average Score: {stats['average_score']:.2f}/10")
            print(f"   Questions Asked: {stats['total_questions_asked']}")
            print(f"   Corrections Made: {stats['total_corrections']}")
            print(f"   Log File: {result['log_file']}")
        else:
            print(f"Training failed: {result.get('error', result.get('message', 'Unknown error'))}")
            
    except KeyboardInterrupt:
        print("\n⏹Training interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

def run_evaluation_only(orchestrator):
    """Run evaluation without training"""
    print("\EVALUATION MODE")
    print("-" * 40)
    
    try:
        num_questions = input("Number of evaluation questions (default 3): ").strip()
        num_questions = int(num_questions) if num_questions else 3
        
        print(f"\nStarting evaluation with {num_questions} questions...")
        result = orchestrator.run_evaluation_only(num_questions)
        
        if result["success"]:
            avg_score = sum(r["evaluation"]["score"] for r in result["results"]) / len(result["results"])
            print(f"\nEvaluation completed!")
            print(f"   Average Score: {avg_score:.2f}/10")
            print(f"   Questions Evaluated: {len(result['results'])}")
        else:
            print(f"Evaluation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")

def interactive_chat_mode(config):
    """Start interactive chat with student AI"""
    print("\nINTERACTIVE CHAT MODE")
    print("-" * 40)
    
    try:
        student = StudentAI(config.student)
        student.interactive_chat()
    except Exception as e:
        print(f"Error starting chat: {e}")

def show_system_config(config):
    """Display system configuration"""
    print("\n SYSTEM CONFIGURATION")
    print("-" * 40)
    
    print("Teacher AI (Groq):")
    print(f"  Model: {config.teacher.model_name}")
    print(f"  Max Tokens: {config.teacher.max_tokens}")
    print(f"  Temperature: {config.teacher.temperature}")
    print(f"  API Key: {'Set' if config.teacher.api_key else 'Missing'}")
    
    print("\nStudent AI:")
    print(f"  Model: {config.student.model_name}")
    print(f"  Device: {config.student.device}")
    print(f"  Max Length: {config.student.max_length}")
    print(f"  Temperature: {config.student.temperature}")
    
    print("\nTraining:")
    print(f"  Dataset: {config.training.dataset_path}")
    print(f"  Output Directory: {config.training.output_dir}")
    print(f"  Batch Size: {config.training.batch_size}")
    
    # Check dataset
    if os.path.exists(config.training.dataset_path):
        data_manager = DataManager(config.training.dataset_path)
        info = data_manager.get_dataset_info()
        print(f"  Dataset Status: Loaded ({info['total_questions']} questions)")
    else:
        print(f"  Dataset Status: File not found")

def test_dataset_manager(config):
    """Test dataset manager functionality"""
    print("\nDATASET MANAGER TEST")
    print("-" * 40)
    
    try:
        data_manager = DataManager(config.training.dataset_path)
        info = data_manager.get_dataset_info()
        
        print(f"Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\nSample Questions:")
        for i in range(min(3, len(data_manager.qa_pairs))):
            qa = data_manager.qa_pairs[i]
            print(f"  Q{i+1}: {qa.question}")
            print(f"  A{i+1}: {qa.answer}")
            print()
            
        # Test random question
        random_qa = data_manager.get_random_question()
        if random_qa:
            print(f"Random Question:")
            print(f"  Q: {random_qa.question}")
            print(f"  A: {random_qa.answer}")
            
    except Exception as e:
        print(f"Error testing dataset: {e}")

def generate_performance_report(orchestrator):
    """Generate performance report"""
    print("\nPERFORMANCE REPORT")
    print("-" * 40)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(orchestrator.config)
        
        # Get test data
        test_data = orchestrator.data_manager.get_batch_questions(5)
        if not test_data:
            print("No test data available")
            return
        
        print("Generating comprehensive evaluation report...")
        result = evaluator.run_comprehensive_evaluation(
            orchestrator.student, 
            orchestrator.teacher, 
            test_data
        )
        
        print(f"\nReport generated!")
        print(f"   Report File: {result['report_file']}")
        if result['chart_file']:
            print(f"   Chart File: {result['chart_file']}")
        
        # Display summary
        if result['baseline_results']:
            stats = result['baseline_results']['statistics']
            print(f"\n📊 Performance Summary:")
            print(f"   Average Score: {stats['average_score']:.2f}/10")
            print(f"   Questions: {stats['total_questions']}")
            print(f"   Score Range: {stats['min_score']:.1f} - {stats['max_score']:.1f}")
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")

def show_help():
    """Show help and documentation"""
    help_text = """
    📖 AI TRAINING SYSTEM - HELP
    ═══════════════════════════════════════════════════════════════
    
    OVERVIEW:
    This system uses a teacher-student AI paradigm where a large model 
    (Teacher) evaluates and guides a smaller model (Student) to improve
    its responses through iterative training.
    
    COMPONENTS:
    • Teacher AI: Uses Groq API with Llama-3-70B for evaluation
    • Student AI: Local small model (e.g., DialoGPT, TinyLlama)
    • Dataset: CSV file with question-answer pairs
    • Orchestrator: Manages training flow and sessions
    • Evaluator: Provides performance analysis and reporting
    
    SETUP REQUIREMENTS:
    1. Set GROQ_API_KEY environment variable
    2. Install requirements: pip install -r requirements.txt
    3. Ensure dataset exists: data/sample_qa_dataset.csv
    
    TRAINING PROCESS:
    1. Student AI attempts to answer questions from dataset
    2. Teacher AI evaluates answers and provides feedback
    3. Results are logged for analysis and improvement tracking
    4. Multiple training cycles can be run progressively
    
    EVALUATION:
    • Baseline evaluation establishes initial performance
    • Post-training evaluation measures improvement
    • Comprehensive reports with charts are generated
    
    FILES STRUCTURE:
    • config/ai_config.py - AI model configurations
    • src/teacher.py - Teacher AI implementation
    • src/student.py - Student AI implementation
    • src/orchestrator.py - Training orchestration
    • src/evaluator.py - Performance evaluation
    • data/ - Dataset files
    • logs/ - Training logs and results
    
    TIPS:
    • Start with small training sessions (3-5 questions)
    • Monitor student responses for quality improvement
    • Use evaluation mode to test without training
    • Review logs to understand learning patterns
    """
    print(help_text)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="AI Training System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner display")
    
    args = parser.parse_args()
    
    if not args.no_banner:
        print_banner()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = get_default_config()
    
    if args.dataset:
        config.training.dataset_path = args.dataset
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Configuration validation failed")
        print("Please check your GROQ_API_KEY and other settings")
        return 1
    
    # Initialize orchestrator
    try:
        print("🚀 Initializing AI Training System...")
        orchestrator = TrainingOrchestrator(config)
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return 1
    
    # Main application loop
    while True:
        try:
            show_main_menu()
            choice = input("Select option (1-9): ").strip()
            
            if choice == "1":
                run_training_session(orchestrator)
            elif choice == "2":
                run_evaluation_only(orchestrator)
            elif choice == "3":
                interactive_chat_mode(config)
            elif choice == "4":
                orchestrator.interactive_training_session()
            elif choice == "5":
                generate_performance_report(orchestrator)
            elif choice == "6":
                show_system_config(config)
            elif choice == "7":
                test_dataset_manager(config)
            elif choice == "8":
                show_help()
            elif choice == "9":
                print("\n👋 Thank you for using AI Training System!")
                break
            else:
                print("❌ Invalid option. Please select 1-9.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")
    
    return 0

if __name__ == "__main__":
    exit(main())