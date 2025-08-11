"""
Training System Orchestrator
Manages the training flow between teacher and student
"""
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from src.teacher import TeacherAI
from src.student import StudentAI
from src.utils.data_manager import DataManager
from src.utils.json_manager import JSONManager

class TrainingOrchestrator:
    """Orchestrates the training process between teacher and student AI"""
    
    def __init__(self, config):
        self.config = config
        self.session_id = self._generate_session_id()
        
        # Initialize components
        self.teacher = None
        self.student = None
        self.data_manager = None
        self.json_manager = JSONManager(config.training.output_dir)
        
        # Training state
        self.current_epoch = 0
        self.training_active = False
        self.training_stats = {
            "total_questions_asked": 0,
            "total_corrections": 0,
            "average_score": 0.0,
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
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise
    
    def run_training_cycle(self, num_questions: int = 10) -> Dict[str, Any]:
        """Run a complete training cycle"""
        if not self._validate_components():
            raise Exception("Components not properly initialized")
        
        print(f"\n=== Starting Training Cycle ===")
        print(f"Session ID: {self.session_id}")
        print(f"Questions to ask: {num_questions}")
        print(f"Dataset: {self.data_manager.get_dataset_info()['total_questions']} questions available")
        
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
                
                # Student answers
                student_response = self.student.generate_response(qa_pair.question)
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
                
                # Store correction
                correction_file = self.json_manager.save_correction(
                    question=qa_pair.question,
                    student_answer=student_response,
                    teacher_feedback=feedback,
                    score=score,
                    session_id=self.session_id
                )
                
                # Update statistics
                self.training_stats["total_questions_asked"] += 1
                if score < 8:  # Consider it a correction if score is below 8
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
                    "correction_file": correction_file
                })
                
                # Small delay to prevent overwhelming APIs
                time.sleep(1)
            
            # Calculate final statistics
            if cycle_results:
                self.training_stats["average_score"] = total_score / len(cycle_results)
            
            self.training_stats["end_time"] = time.time()
            self.training_active = False
            
            # Save training log
            log_data = {
                "session_id": self.session_id,
                "cycle_results": cycle_results,
                "training_stats": self.training_stats,
                "config_info": {
                    "teacher_model": self.config.teacher.model_name,
                    "student_model": self.config.student.model_name,
                    "dataset_path": self.config.training.dataset_path
                }
            }
            
            log_file = self.json_manager.save_training_log(log_data, self.session_id)
            
            print(f"\n=== Training Cycle Complete ===")
            print(f"Questions asked: {len(cycle_results)}")
            print(f"Average score: {self.training_stats['average_score']:.2f}/10")
            print(f"Corrections made: {self.training_stats['total_corrections']}")
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
            return {"success": False, "message": "Training interrupted"}
            
        except Exception as e:
            print(f"\nError during training cycle: {e}")
            self.training_active = False
            return {"success": False, "error": str(e)}
    
    def run_evaluation_only(self, num_questions: int = 5) -> Dict[str, Any]:
        """Run evaluation without training (just testing)"""
        print(f"\n=== Running Evaluation Mode ===")
        
        evaluation_results = []
        
        try:
            for i in range(num_questions):
                qa_pair = self.data_manager.get_random_question()
                if not qa_pair:
                    break
                
                print(f"\nEvaluation {i+1}/{num_questions}")
                print(f"Q: {qa_pair.question}")
                
                student_response = self.student.generate_response(qa_pair.question)
                print(f"Student: {student_response}")
                
                evaluation = self.teacher.evaluate_answer(
                    question=qa_pair.question,
                    correct_answer=qa_pair.answer,
                    student_answer=student_response
                )
                
                print(f"Score: {evaluation.get('score', 0)}/10")
                
                evaluation_results.append({
                    "question": qa_pair.question,
                    "expected_answer": qa_pair.answer,
                    "student_answer": student_response,
                    "evaluation": evaluation
                })
            
            # Save evaluation results
            eval_file = self.json_manager.save_evaluation_results({
                "evaluation_type": "baseline_test",
                "total_questions": len(evaluation_results),
                "results": evaluation_results,
                "average_score": sum(r["evaluation"]["score"] for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
            }, self.session_id)
            
            return {
                "success": True,
                "results": evaluation_results,
                "evaluation_file": eval_file
            }
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {"success": False, "error": str(e)}
    
    def interactive_training_session(self):
        """Run interactive training session"""
        print(f"\n=== Interactive Training Session ===")
        print("Commands: 'ask' - ask question, 'eval' - evaluation mode, 'stats' - show stats, 'quit' - exit")
        
        while True:
            try:
                command = input("\nTraining> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                
                elif command == 'ask':
                    num_questions = input("How many questions? (default 1): ").strip()
                    try:
                        num_questions = int(num_questions) if num_questions else 1
                    except ValueError:
                        num_questions = 1
                    
                    result = self.run_training_cycle(num_questions)
                    if result["success"]:
                        print(f"Training completed! Average score: {result['statistics']['average_score']:.2f}")
                
                elif command == 'eval':
                    num_questions = input("How many evaluation questions? (default 3): ").strip()
                    try:
                        num_questions = int(num_questions) if num_questions else 3
                    except ValueError:
                        num_questions = 3
                    
                    result = self.run_evaluation_only(num_questions)
                    if result["success"]:
                        avg_score = sum(r["evaluation"]["score"] for r in result["results"]) / len(result["results"])
                        print(f"Evaluation completed! Average score: {avg_score:.2f}")
                
                elif command == 'stats':
                    self.show_session_statistics()
                
                elif command == 'chat':
                    print("Entering student chat mode...")
                    self.student.interactive_chat()
                
                elif command == 'help':
                    print("Available commands:")
                    print("  ask - Run training cycle")
                    print("  eval - Run evaluation only")
                    print("  chat - Chat with student")
                    print("  stats - Show statistics")
                    print("  help - Show this help")
                    print("  quit - Exit")
                
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nExiting interactive session...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_session_statistics(self):
        """Show current session statistics"""
        print(f"\n=== Session Statistics ===")
        print(f"Session ID: {self.session_id}")
        print(f"Training Active: {self.training_active}")
        print(f"Questions Asked: {self.training_stats['total_questions_asked']}")
        print(f"Corrections Made: {self.training_stats['total_corrections']}")
        print(f"Average Score: {self.training_stats['average_score']:.2f}/10")
        
        if self.training_stats['start_time']:
            duration = (self.training_stats['end_time'] or time.time()) - self.training_stats['start_time']
            print(f"Session Duration: {duration:.2f} seconds")
        
        # Show teacher statistics
        teacher_stats = self.teacher.get_evaluation_statistics()
        if teacher_stats.get("total_evaluations", 0) > 0:
            print(f"\n--- Teacher Statistics ---")
            print(f"Total Evaluations: {teacher_stats['total_evaluations']}")
            print(f"Average Score: {teacher_stats['average_score']:.2f}")
            print("Score Distribution:")
            for category, count in teacher_stats['score_distribution'].items():
                print(f"  {category}: {count}")
        
        # Show student info
        student_info = self.student.get_model_info()
        print(f"\n--- Student Model Info ---")
        print(f"Model: {student_info['model_name']}")
        print(f"Device: {student_info['device']}")
        print(f"Conversations: {student_info['conversation_length']}")
    
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
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information"""
        return {
            "session_id": self.session_id,
            "training_active": self.training_active,
            "statistics": self.training_stats,
            "teacher_stats": self.teacher.get_evaluation_statistics() if self.teacher else {},
            "student_info": self.student.get_model_info() if self.student else {},
            "dataset_info": self.data_manager.get_dataset_info() if self.data_manager else {}
        }