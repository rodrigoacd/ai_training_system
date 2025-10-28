"""
Enhanced Training Orchestrator with FAISS Memory Integration
Demonstrates how to integrate the memory module into the training feedback loop.
"""
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.teacher import TeacherAI
from src.student import StudentAI
from src.utils.data_manager import DataManager
from src.utils.json_manager import JSONManager
from src.utils.memory_module import FAISSMemoryModule


class MemoryEnhancedOrchestrator:
    """
    Enhanced orchestrator with FAISS memory for learning from past interactions.
    
    The memory module enables the Student AI to:
    1. Retrieve similar past Q&A interactions when answering new questions
    2. Learn from previous mistakes and successful answers
    3. Improve responses by leveraging historical feedback
    """
    
    def __init__(self, config, use_memory: bool = True):
        self.config = config
        self.session_id = self._generate_session_id()
        
        # Initialize components
        self.teacher = None
        self.student = None
        self.data_manager = None
        self.json_manager = JSONManager(config.training.output_dir)
        
        # Initialize memory module
        self.use_memory = use_memory
        self.memory = None
        if use_memory:
            self.memory = FAISSMemoryModule(
                embedding_model="all-MiniLM-L6-v2",
                index_type="cosine",  # Cosine similarity works well for semantic search
                storage_path="memory"
            )
            self._load_existing_memory()
        
        # Training state
        self.current_epoch = 0
        self.training_active = False
        self.training_stats = {
            "total_questions_asked": 0,
            "total_corrections": 0,
            "average_score": 0.0,
            "memory_retrievals": 0,
            "start_time": None,
            "end_time": None
        }
        
        self._initialize_components()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            print(f"Initializing training session: {self.session_id}")
            
            # Initialize Teacher AI
            print("Loading Teacher AI (Groq)...")
            self.teacher = TeacherAI(self.config.teacher)
            
            # Initialize Student AI
            print("Loading Student AI...")
            self.student = StudentAI(self.config.student)
            
            # Initialize Data Manager
            print("Loading dataset...")
            self.data_manager = DataManager(self.config.training.dataset_path)
            
            print("All components initialized successfully!")
            
            if self.memory:
                print(f"Memory module active with {self.memory.memory_count} stored interactions")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise
    
    def _load_existing_memory(self):
        """Load existing memory from disk if available"""
        memory_path = self.memory.storage_path
        
        # Find most recent memory files
        index_files = sorted(memory_path.glob("memory_index_*.faiss"))
        metadata_files = sorted(memory_path.glob("memory_metadata_*.json"))
        
        if index_files and metadata_files:
            latest_index = str(index_files[-1])
            latest_metadata = str(metadata_files[-1])
            
            print(f"Loading existing memory from disk...")
            if self.memory.load(latest_index, latest_metadata):
                print(f"✓ Loaded {self.memory.memory_count} previous interactions")
            else:
                print("✗ Failed to load existing memory, starting fresh")
    
    def run_training_cycle_with_memory(
        self,
        num_questions: int = 10,
        retrieve_k: int = 3,
        memory_weight: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run training cycle with memory-augmented learning.
        
        Args:
            num_questions: Number of questions to train on
            retrieve_k: Number of similar past interactions to retrieve
            memory_weight: Weight for memory influence (0.0-1.0)
            
        Returns:
            Dictionary with training results and statistics
        """
        if not self._validate_components():
            raise Exception("Components not properly initialized")
        
        print(f"\n=== Starting Memory-Enhanced Training Cycle ===")
        print(f"Session ID: {self.session_id}")
        print(f"Questions: {num_questions}")
        print(f"Memory retrieval: {'Enabled' if self.use_memory else 'Disabled'}")
        
        self.training_active = True
        self.training_stats["start_time"] = time.time()
        
        cycle_results = []
        total_score = 0
        
        try:
            for i in range(num_questions):
                print(f"\n--- Question {i+1}/{num_questions} ---")
                
                # Get question from dataset
                qa_pair = self.data_manager.get_random_question()
                if not qa_pair:
                    print("No more questions available")
                    break
                
                print(f"Question: {qa_pair.question}")
                
                # MEMORY RETRIEVAL: Get similar past interactions
                context = ""
                similar_memories = []
                if self.use_memory and self.memory.memory_count > 0:
                    similar_memories = self.memory.retrieve_similar(
                        query=qa_pair.question,
                        top_k=retrieve_k,
                        min_score=7.0,  # Only retrieve high-quality examples
                        exclude_session=self.session_id  # Don't use current session
                    )
                    
                    if similar_memories:
                        context = self.memory.format_context(similar_memories)
                        self.training_stats["memory_retrievals"] += 1
                        print(f"Retrieved {len(similar_memories)} similar past interactions")
                
                # Student answers WITH memory context
                student_response = self.student.generate_response(
                    question=qa_pair.question,
                    context=context if context else None
                )
                print(f"Student Answer: {student_response}")
                
                # Teacher evaluates
                evaluation = self.teacher.evaluate_answer(
                    question=qa_pair.question,
                    correct_answer=qa_pair.answer,
                    student_answer=student_response
                )
                
                score = evaluation.get("score", 0)
                feedback = evaluation.get("feedback", "")
                
                print(f"Teacher Score: {score}/10")
                print(f"Teacher Feedback: {feedback}")
                
                # MEMORY STORAGE: Add this interaction to memory
                if self.use_memory:
                    memory_idx = self.memory.add_interaction(
                        question=qa_pair.question,
                        student_answer=student_response,
                        teacher_feedback=feedback,
                        score=score,
                        session_id=self.session_id
                    )
                    print(f"Stored in memory (index: {memory_idx})")
                
                # Store correction in JSON
                correction_file = self.json_manager.save_correction(
                    question=qa_pair.question,
                    student_answer=student_response,
                    teacher_feedback=feedback,
                    score=score,
                    session_id=self.session_id
                )
                
                # Update statistics
                self.training_stats["total_questions_asked"] += 1
                if score < 8:
                    self.training_stats["total_corrections"] += 1
                
                total_score += score
                
                # Store cycle result
                cycle_results.append({
                    "question_number": i + 1,
                    "question": qa_pair.question,
                    "correct_answer": qa_pair.answer,
                    "student_answer": student_response,
                    "score": score,
                    "feedback": feedback,
                    "memory_used": len(similar_memories) > 0,
                    "memory_count": len(similar_memories),
                    "correction_file": correction_file
                })
                
                # Small delay
                time.sleep(1)
            
            # Calculate statistics
            if cycle_results:
                self.training_stats["average_score"] = total_score / len(cycle_results)
            
            self.training_stats["end_time"] = time.time()
            self.training_active = False
            
            # Save memory to disk
            if self.use_memory:
                self.memory.save()
            
            # Save training log
            log_data = {
                "session_id": self.session_id,
                "cycle_results": cycle_results,
                "training_stats": self.training_stats,
                "memory_stats": self.memory.get_statistics() if self.memory else {},
                "config_info": {
                    "teacher_model": self.config.teacher.model_name,
                    "student_model": self.config.student.model_name,
                    "dataset_path": self.config.training.dataset_path,
                    "memory_enabled": self.use_memory
                }
            }
            
            log_file = self.json_manager.save_training_log(log_data, self.session_id)
            
            print(f"\n=== Training Cycle Complete ===")
            print(f"Questions asked: {len(cycle_results)}")
            print(f"Average score: {self.training_stats['average_score']:.2f}/10")
            print(f"Corrections made: {self.training_stats['total_corrections']}")
            print(f"Memory retrievals: {self.training_stats['memory_retrievals']}")
            if self.memory:
                print(f"Total memories stored: {self.memory.memory_count}")
            print(f"Training log saved: {log_file}")
            
            return {
                "success": True,
                "session_id": self.session_id,
                "results": cycle_results,
                "statistics": self.training_stats,
                "log_file": log_file
            }
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.training_active = False
            
            # Save memory even if interrupted
            if self.use_memory:
                self.memory.save()
            
            return {"success": False, "message": "Training interrupted"}
            
        except Exception as e:
            print(f"\nError during training cycle: {e}")
            self.training_active = False
            return {"success": False, "error": str(e)}
    
    def perform_memory_maintenance(
        self,
        max_entries: int = 1000,
        min_score: float = 6.0,
        keep_recent: int = 100
    ):
        """
        Perform periodic memory maintenance.
        
        This should be called periodically to:
        - Remove low-quality interactions
        - Limit total memory size
        - Improve retrieval performance
        
        Args:
            max_entries: Maximum total entries to keep
            min_score: Remove entries below this score
            keep_recent: Always keep this many recent entries
        """
        if not self.use_memory:
            print("Memory module not enabled")
            return
        
        print("\n=== Memory Maintenance ===")
        
        before_count = self.memory.memory_count
        stats_before = self.memory.get_statistics()
        
        print(f"Before: {before_count} entries")
        print(f"Average score: {stats_before['average_score']:.2f}")
        
        # Prune memory
        removed = self.memory.prune_memory(
            max_entries=max_entries,
            min_score=min_score,
            keep_recent=keep_recent
        )
        
        stats_after = self.memory.get_statistics()
        
        print(f"After: {self.memory.memory_count} entries")
        print(f"Removed: {removed} entries")
        print(f"New average score: {stats_after['average_score']:.2f}")
        
        # Save pruned memory
        if removed > 0:
            self.memory.save("memory_pruned")
    
    def show_memory_statistics(self):
        """Display detailed memory statistics"""
        if not self.use_memory:
            print("Memory module not enabled")
            return
        
        print("\n=== Memory Statistics ===")
        stats = self.memory.get_statistics()
        
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    def _validate_components(self) -> bool:
        """Validate that all components are properly initialized"""
        if not self.teacher:
            print("Error: Teacher AI not initialized")
            return False
        
        if not self.student:
            print("Error: Student AI not initialized")
            return False
        
        if not self.data_manager or not self.data_manager.qa_pairs:
            print("Error: Data Manager not initialized or no data loaded")
            return False
        
        return True


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example of how to use the memory-enhanced orchestrator.
    """
    from config.ai_config import get_default_config
    
    # Load configuration
    config = get_default_config()
    
    # Initialize orchestrator with memory
    orchestrator = MemoryEnhancedOrchestrator(config, use_memory=True)
    
    # Run training cycles with memory
    print("Running training cycle 1...")
    result1 = orchestrator.run_training_cycle_with_memory(
        num_questions=5,
        retrieve_k=3,  # Retrieve 3 similar past interactions
        memory_weight=0.7
    )
    
    # The memory now contains 5 interactions
    orchestrator.show_memory_statistics()
    
    # Run another cycle - student can now learn from previous interactions
    print("\nRunning training cycle 2...")
    result2 = orchestrator.run_training_cycle_with_memory(
        num_questions=5,
        retrieve_k=3
    )
    
    # Compare scores
    if result1["success"] and result2["success"]:
        score1 = result1["statistics"]["average_score"]
        score2 = result2["statistics"]["average_score"]
        improvement = score2 - score1
        
        print(f"\n=== Performance Comparison ===")
        print(f"Cycle 1 average score: {score1:.2f}/10")
        print(f"Cycle 2 average score: {score2:.2f}/10")
        print(f"Improvement: {improvement:+.2f} points")