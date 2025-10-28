"""
AI Training System - Main Application (with Memory Support)
Entry point for the AI training and evaluation system
"""
import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.ai_config import get_default_config, validate_config, setup_environment
from src.evaluator import ModelEvaluator
from src.student import StudentAI
from src.utils.data_manager import DataManager

# Import both orchestrator versions
try:
    from src.orchestrator_with_memory import MemoryEnhancedOrchestrator
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("Warning: Memory module not available, using standard orchestrator")

from src.orchestrator import TrainingOrchestrator

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    AI TRAINING SYSTEM                        ║
    ║              Teacher-Student AI Learning Platform            ║
    ║                   🧠 WITH MEMORY MODULE 🧠                   ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Components:
    • Teacher AI: Groq Llama-3-70B (Evaluation & Guidance)
    • Student AI: Small Local Model (Learning & Responses)
    • Memory System: FAISS Semantic Search (Progressive Learning)
    • Training Data: Q&A Dataset with Feedback Loop
    
    """
    print(banner)

def show_main_menu(memory_enabled: bool = False):
    """Display main menu options"""
    memory_status = "🟢 ENABLED" if memory_enabled else "🔴 DISABLED"
    
    menu = f"""
    ═══════════════ MAIN MENU ═══════════════
    Memory System: {memory_status}
    
    1. 🎯 Run Training Session (Standard)
    2. 🧠 Run Training with Memory Enhancement
    3. 📊 Run Evaluation Only  
    4. 💬 Interactive Chat with Student
    5. 🎮 Interactive Training Mode
    6. 📈 Generate Performance Report
    7. 🗄️  Memory Management
    8. ⚙️  System Configuration
    9. 🔍 Test Dataset Manager
    10. ❓ Help & Documentation
    11. 🚪 Exit
    
    ════════════════════════════════════════════
    """
    print(menu)

def run_training_session(orchestrator, use_memory: bool = False):
    """Run a training session"""
    training_type = "Memory-Enhanced Training" if use_memory else "Standard Training"
    print(f"\n{training_type.upper()}")
    print("-" * 40)
    
    try:
        num_questions = input("Number of training questions (default 5): ").strip()
        num_questions = int(num_questions) if num_questions else 5
        
        if use_memory and hasattr(orchestrator, 'run_training_cycle_with_memory'):
            # Memory-enhanced training
            retrieve_k = input("Number of similar memories to retrieve (default 3): ").strip()
            retrieve_k = int(retrieve_k) if retrieve_k else 3
            
            print(f"\nStarting memory-enhanced training with {num_questions} questions...")
            print(f"Retrieving {retrieve_k} similar past interactions per question...")
            
            result = orchestrator.run_training_cycle_with_memory(
                num_questions=num_questions,
                retrieve_k=retrieve_k
            )
        else:
            # Standard training
            print(f"\nStarting standard training with {num_questions} questions...")
            result = orchestrator.run_training_cycle(num_questions)
        
        if result["success"]:
            stats = result["statistics"]
            print(f"\n✅ Training completed successfully!")
            print(f"   Average Score: {stats['average_score']:.2f}/10")
            print(f"   Questions Asked: {stats['total_questions_asked']}")
            print(f"   Corrections Made: {stats['total_corrections']}")
            
            if use_memory and 'memory_retrievals' in stats:
                print(f"   Memory Retrievals: {stats['memory_retrievals']}")
                if hasattr(orchestrator, 'memory'):
                    print(f"   Total Memories Stored: {orchestrator.memory.memory_count}")
            
            print(f"   Log File: {result['log_file']}")
        else:
            print(f"❌ Training failed: {result.get('error', result.get('message', 'Unknown error'))}")
            
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_evaluation_only(orchestrator):
    """Run evaluation without training"""
    print("\nEVALUATION MODE")
    print("-" * 40)
    
    try:
        num_questions = input("Number of evaluation questions (default 3): ").strip()
        num_questions = int(num_questions) if num_questions else 3
        
        print(f"\nStarting evaluation with {num_questions} questions...")
        result = orchestrator.run_evaluation_only(num_questions)
        
        if result["success"]:
            avg_score = sum(r["evaluation"]["score"] for r in result["results"]) / len(result["results"])
            print(f"\n✅ Evaluation completed!")
            print(f"   Average Score: {avg_score:.2f}/10")
            print(f"   Questions Evaluated: {len(result['results'])}")
        else:
            print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_chat_mode(config):
    """Start interactive chat with student AI"""
    print("\nINTERACTIVE CHAT MODE")
    print("-" * 40)
    
    try:
        student = StudentAI(config.student)
        student.interactive_chat()
    except Exception as e:
        print(f"❌ Error starting chat: {e}")

def memory_management_menu(orchestrator):
    """Memory management submenu"""
    if not hasattr(orchestrator, 'memory') or orchestrator.memory is None:
        print("\n❌ Memory module not available")
        return
    
    while True:
        print("\n" + "="*50)
        print("MEMORY MANAGEMENT")
        print("="*50)
        print("\n1. 📊 Show Memory Statistics")
        print("2. 🧹 Prune Memory (Remove Low Quality)")
        print("3. 💾 Save Memory to Disk")
        print("4. 📂 Load Memory from Disk")
        print("5. 🗑️  Clear All Memory")
        print("6. 🔍 Search Memory")
        print("7. 📈 Memory Health Check")
        print("8. ⬅️  Back to Main Menu")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            show_memory_statistics(orchestrator)
        elif choice == "2":
            prune_memory(orchestrator)
        elif choice == "3":
            save_memory(orchestrator)
        elif choice == "4":
            load_memory(orchestrator)
        elif choice == "5":
            clear_memory(orchestrator)
        elif choice == "6":
            search_memory(orchestrator)
        elif choice == "7":
            memory_health_check(orchestrator)
        elif choice == "8":
            break
        else:
            print("❌ Invalid option")
        
        if choice != "8":
            input("\nPress Enter to continue...")

def show_memory_statistics(orchestrator):
    """Display memory statistics"""
    print("\n📊 MEMORY STATISTICS")
    print("-" * 40)
    
    stats = orchestrator.memory.get_statistics()
    
    print(f"Total Memories: {stats['total_memories']}")
    print(f"Index Type: {stats['index_type']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Embedding Model: {stats['embedding_model']}")
    
    if stats['total_memories'] > 0:
        print(f"\nQuality Metrics:")
        print(f"  Average Score: {stats['average_score']:.2f}/10")
        print(f"  Score Range: {stats['score_range'][0]:.1f} - {stats['score_range'][1]:.1f}")
        print(f"  Unique Sessions: {stats['unique_sessions']}")
        print(f"\nTime Range:")
        print(f"  Oldest Entry: {stats['oldest_entry']}")
        print(f"  Newest Entry: {stats['newest_entry']}")

def prune_memory(orchestrator):
    """Prune low-quality memories"""
    print("\n🧹 MEMORY PRUNING")
    print("-" * 40)
    
    try:
        max_entries = input("Max entries to keep (default 1000): ").strip()
        max_entries = int(max_entries) if max_entries else 1000
        
        min_score = input("Minimum score threshold (default 6.0): ").strip()
        min_score = float(min_score) if min_score else 6.0
        
        keep_recent = input("Always keep N recent entries (default 100): ").strip()
        keep_recent = int(keep_recent) if keep_recent else 100
        
        print(f"\nPruning memory...")
        orchestrator.perform_memory_maintenance(
            max_entries=max_entries,
            min_score=min_score,
            keep_recent=keep_recent
        )
        print("✅ Memory pruned successfully")
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error pruning memory: {e}")

def save_memory(orchestrator):
    """Save memory to disk"""
    print("\n💾 SAVE MEMORY")
    print("-" * 40)
    
    try:
        prefix = input("Filename prefix (default 'memory'): ").strip()
        prefix = prefix if prefix else "memory"
        
        index_path, metadata_path = orchestrator.memory.save(prefix)
        print(f"\n✅ Memory saved successfully!")
        print(f"   Index: {index_path}")
        print(f"   Metadata: {metadata_path}")
        
    except Exception as e:
        print(f"❌ Error saving memory: {e}")

def load_memory(orchestrator):
    """Load memory from disk"""
    print("\n📂 LOAD MEMORY")
    print("-" * 40)
    
    try:
        memory_path = Path(orchestrator.memory.storage_path)
        
        # List available memory files
        index_files = sorted(memory_path.glob("memory_index_*.faiss"))
        
        if not index_files:
            print("❌ No memory files found")
            return
        
        print("\nAvailable memory files:")
        for i, file in enumerate(index_files, 1):
            print(f"  {i}. {file.name}")
        
        choice = input(f"\nSelect file (1-{len(index_files)}) or Enter for latest: ").strip()
        
        if choice:
            idx = int(choice) - 1
            if 0 <= idx < len(index_files):
                index_file = index_files[idx]
            else:
                print("❌ Invalid selection")
                return
        else:
            index_file = index_files[-1]
        
        # Find corresponding metadata
        metadata_file = index_file.parent / index_file.name.replace("_index_", "_metadata_").replace(".faiss", ".json")
        
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return
        
        print(f"\nLoading from: {index_file.name}")
        if orchestrator.memory.load(str(index_file), str(metadata_file)):
            print(f"✅ Memory loaded successfully!")
            print(f"   Entries: {orchestrator.memory.memory_count}")
        else:
            print("❌ Failed to load memory")
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error loading memory: {e}")

def clear_memory(orchestrator):
    """Clear all memory"""
    print("\n🗑️  CLEAR MEMORY")
    print("-" * 40)
    
    confirm = input("⚠️  This will delete all stored memories. Continue? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        orchestrator.memory.clear_memory()
        print("✅ All memory cleared")
    else:
        print("❌ Operation cancelled")

def search_memory(orchestrator):
    """Search memory interactively"""
    print("\n🔍 SEARCH MEMORY")
    print("-" * 40)
    
    if orchestrator.memory.memory_count == 0:
        print("❌ No memories stored yet")
        return
    
    query = input("Enter search query: ").strip()
    if not query:
        return
    
    try:
        top_k = input("Number of results (default 5): ").strip()
        top_k = int(top_k) if top_k else 5
        
        print(f"\nSearching for: '{query}'...")
        similar = orchestrator.memory.retrieve_similar(query, top_k=top_k)
        
        if similar:
            print(f"\n✅ Found {len(similar)} similar memories:")
            for i, (memory, similarity) in enumerate(similar, 1):
                print(f"\n--- Result {i} (Similarity: {similarity:.3f}, Score: {memory.score}/10) ---")
                print(f"Question: {memory.question}")
                print(f"Answer: {memory.student_answer}")
                print(f"Feedback: {memory.teacher_feedback}")
                print(f"Session: {memory.session_id}")
        else:
            print("❌ No similar memories found")
        
    except Exception as e:
        print(f"❌ Error searching: {e}")

def memory_health_check(orchestrator):
    """Perform memory health check"""
    print("\n📈 MEMORY HEALTH CHECK")
    print("-" * 40)
    
    stats = orchestrator.memory.get_statistics()
    
    if stats['total_memories'] == 0:
        print("⚠️  No memories stored yet")
        return
    
    print("Analyzing memory health...\n")
    
    # Size check
    if stats['total_memories'] > 1000:
        print("⚠️  WARNING: Memory size exceeds 1000 entries")
        print(f"   Current: {stats['total_memories']} entries")
        print("   Recommendation: Run pruning")
    else:
        print(f"✅ Memory size OK: {stats['total_memories']} entries")
    
    # Quality check
    if stats['average_score'] < 6.0:
        print("\n⚠️  WARNING: Low average quality")
        print(f"   Average score: {stats['average_score']:.2f}/10")
        print("   Recommendation: Increase min_score threshold during retrieval")
    else:
        print(f"\n✅ Quality OK: {stats['average_score']:.2f}/10 average score")
    
    # Age check
    from datetime import datetime
    try:
        oldest = datetime.fromisoformat(stats['oldest_entry'])
        age_days = (datetime.now() - oldest).days
        
        if age_days > 30:
            print(f"\n⚠️  INFO: Oldest entry is {age_days} days old")
            print("   Consider refreshing memory if training data has changed")
        else:
            print(f"\n✅ Freshness OK: {age_days} days old")
    except:
        pass
    
    # Diversity check
    if stats['unique_sessions'] < 2:
        print(f"\n⚠️  INFO: Limited diversity")
        print(f"   Only {stats['unique_sessions']} training session(s)")
        print("   Recommendation: Run more training sessions for better coverage")
    else:
        print(f"\n✅ Diversity OK: {stats['unique_sessions']} different sessions")

def show_system_config(config, orchestrator=None):
    """Display system configuration"""
    print("\n⚙️  SYSTEM CONFIGURATION")
    print("-" * 40)
    
    print("Teacher AI (Groq):")
    print(f"  Model: {config.teacher.model_name}")
    print(f"  Max Tokens: {config.teacher.max_tokens}")
    print(f"  Temperature: {config.teacher.temperature}")
    print(f"  API Key: {'✅ Set' if config.teacher.api_key else '❌ Missing'}")
    
    print("\nStudent AI:")
    print(f"  Model: {config.student.model_name}")
    print(f"  Device: {config.student.device}")
    print(f"  Max Length: {config.student.max_length}")
    print(f"  Temperature: {config.student.temperature}")
    
    print("\nMemory System:")
    if hasattr(config, 'memory'):
        print(f"  Status: {'✅ Enabled' if config.memory.enabled else '❌ Disabled'}")
        print(f"  Embedding Model: {config.memory.embedding_model}")
        print(f"  Index Type: {config.memory.index_type}")
        print(f"  Max Entries: {config.memory.max_entries}")
        print(f"  Retrieve K: {config.memory.retrieve_k}")
        print(f"  Min Score: {config.memory.min_score_threshold}")
        
        if orchestrator and hasattr(orchestrator, 'memory'):
            print(f"  Current Entries: {orchestrator.memory.memory_count}")
    else:
        print(f"  Status: ❌ Not configured")
    
    print("\nTraining:")
    print(f"  Dataset: {config.training.dataset_path}")
    print(f"  Output Directory: {config.training.output_dir}")
    print(f"  Batch Size: {config.training.batch_size}")
    
    # Check dataset
    if os.path.exists(config.training.dataset_path):
        data_manager = DataManager(config.training.dataset_path)
        info = data_manager.get_dataset_info()
        print(f"  Dataset Status: ✅ Loaded ({info['total_questions']} questions)")
    else:
        print(f"  Dataset Status: ❌ File not found")

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
        print(f"❌ Error testing dataset: {e}")

def generate_performance_report(orchestrator):
    """Generate performance report"""
    print("\n📈 PERFORMANCE REPORT")
    print("-" * 40)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(orchestrator.config)
        
        # Get test data
        test_data = orchestrator.data_manager.get_batch_questions(5)
        if not test_data:
            print("❌ No test data available")
            return
        
        print("Generating comprehensive evaluation report...")
        result = evaluator.run_comprehensive_evaluation(
            orchestrator.student, 
            orchestrator.teacher, 
            test_data
        )
        
        print(f"\n✅ Report generated!")
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
    This system uses a teacher-student AI paradigm with memory enhancement.
    A large model (Teacher) evaluates and guides a smaller model (Student)
    while a FAISS-based memory system enables progressive learning.
    
    COMPONENTS:
    • Teacher AI: Uses Groq API with Llama-3-70B for evaluation
    • Student AI: Local small model (e.g., DialoGPT, TinyLlama)
    • Memory System: FAISS semantic search for past interactions
    • Dataset: CSV file with question-answer pairs
    • Orchestrator: Manages training flow and sessions
    • Evaluator: Provides performance analysis and reporting
    
    MEMORY SYSTEM:
    The memory module stores past Q&A interactions and retrieves similar
    examples to help the Student learn from previous experiences.
    
    Features:
    - Semantic similarity search using embeddings
    - Stores question-answer-feedback triples
    - Retrieves relevant past interactions
    - Prunes low-quality memories automatically
    - Persists between training sessions
    
    SETUP REQUIREMENTS:
    1. Set GROQ_API_KEY environment variable
    2. Install requirements: pip install -r requirements.txt
    3. Ensure dataset exists: data/sample_qa_dataset.csv
    4. Memory directory will be created automatically
    
    TRAINING PROCESS (WITH MEMORY):
    1. Student receives a question
    2. Memory system retrieves similar past interactions
    3. Student answers using context from memory
    4. Teacher evaluates and provides feedback
    5. Interaction is stored in memory for future use
    6. Process repeats, enabling progressive improvement
    
    TRAINING PROCESS (WITHOUT MEMORY):
    1. Student attempts to answer questions from dataset
    2. Teacher evaluates answers and provides feedback
    3. Results are logged for analysis
    4. No learning between questions in same session
    
    MEMORY MANAGEMENT:
    • View statistics to monitor memory health
    • Prune regularly to maintain performance
    • Save/load to preserve learning across restarts
    • Search to find specific past interactions
    
    BEST PRACTICES:
    • Start with 5-10 questions per session
    • Use memory enhancement for progressive learning
    • Prune memory after every 50-100 questions
    • Keep max 1000 high-quality memories (score ≥ 7)
    • Monitor average scores to track improvement
    • Save memory after successful training sessions
    
    FILES STRUCTURE:
    • config/ai_config.py - AI and memory configurations
    • src/teacher.py - Teacher AI implementation
    • src/student.py - Student AI implementation
    • src/orchestrator.py - Standard training orchestration
    • src/orchestrator_with_memory.py - Memory-enhanced training
    • src/utils/memory_module.py - FAISS memory system
    • data/ - Dataset files
    • logs/ - Training logs and results
    • memory/ - FAISS indices and metadata
    
    For more information, see README.md
    """
    print(help_text)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="AI Training System with Memory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner display")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory module")
    parser.add_argument("--memory-only", action="store_true", help="Force memory usage")
    
    args = parser.parse_args()
    
    if not args.no_banner:
        print_banner()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = get_default_config()
    
    if args.dataset:
        config.training.dataset_path = args.dataset
    
    # Override memory setting if specified
    if args.no_memory:
        config.memory.enabled = False
    elif args.memory_only:
        config.memory.enabled = True
    
    # Validate configuration
    if not validate_config(config):
        print("❌ Configuration validation failed")
        print("Please check your GROQ_API_KEY and other settings")
        return 1
    
    # Determine which orchestrator to use
    use_memory_orchestrator = config.memory.enabled and MEMORY_AVAILABLE
    
    if config.memory.enabled and not MEMORY_AVAILABLE:
        print("⚠️  Warning: Memory module requested but not available")
        print("   Falling back to standard orchestrator")
        print("   Install dependencies: pip install sentence-transformers faiss-cpu")
        use_memory_orchestrator = False
    
    # Initialize orchestrator
    try:
        print(f"🚀 Initializing AI Training System...")
        
        if use_memory_orchestrator:
            orchestrator = MemoryEnhancedOrchestrator(config, use_memory=True)
            print("✅ System initialized with Memory Enhancement!")
        else:
            orchestrator = TrainingOrchestrator(config)
            print("✅ System initialized (Standard Mode)")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Main application loop
    while True:
        try:
            show_main_menu(memory_enabled=use_memory_orchestrator)
            choice = input("Select option (1-11): ").strip()
            
            if choice == "1":
                # Standard training
                run_training_session(orchestrator, use_memory=False)
                
            elif choice == "2":
                # Memory-enhanced training
                if use_memory_orchestrator:
                    run_training_session(orchestrator, use_memory=True)
                else:
                    print("❌ Memory module not available")
                    print("   Install: pip install sentence-transformers faiss-cpu")
                    
            elif choice == "3":
                run_evaluation_only(orchestrator)
                
            elif choice == "4":
                interactive_chat_mode(config)
                
            elif choice == "5":
                orchestrator.interactive_training_session()
                
            elif choice == "6":
                generate_performance_report(orchestrator)
                
            elif choice == "7":
                # Memory management
                if use_memory_orchestrator:
                    memory_management_menu(orchestrator)
                else:
                    print("❌ Memory module not available")
                    
            elif choice == "8":
                show_system_config(config, orchestrator)
                
            elif choice == "9":
                test_dataset_manager(config)
                
            elif choice == "10":
                show_help()
                
            elif choice == "11":
                print("\n👋 Thank you for using AI Training System!")
                
                # Save memory before exit if enabled
                if use_memory_orchestrator and orchestrator.memory.memory_count > 0:
                    save = input("💾 Save memory before exit? (y/n): ").strip().lower()
                    if save in ['y', 'yes']:
                        orchestrator.memory.save("memory_exit")
                        print("✅ Memory saved")
                
                break
            else:
                print("❌ Invalid option. Please select 1-11.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            
            # Save memory on interrupt if possible
            if use_memory_orchestrator and orchestrator.memory.memory_count > 0:
                try:
                    print("💾 Saving memory before exit...")
                    orchestrator.memory.save("memory_interrupt")
                    print("✅ Memory saved")
                except:
                    pass
            break
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            input("Press Enter to continue...")
    
    return 0

if __name__ == "__main__":
    exit(main())