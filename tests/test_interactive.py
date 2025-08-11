"""
Interactive Testing Module
Test student AI responses interactively
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.ai_config import get_default_config, setup_environment
from src.student import StudentAI
from src.teacher import TeacherAI
from src.utils.data_manager import DataManager

class InteractiveTestRunner:
    """Interactive test runner for student AI"""
    
    def __init__(self):
        setup_environment()
        self.config = get_default_config()
        self.student = None
        self.teacher = None
        self.data_manager = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize AI components"""
        try:
            print("🤖 Loading Student AI...")
            self.student = StudentAI(self.config.student)
            
            print("👨‍🏫 Loading Teacher AI...")
            self.teacher = TeacherAI(self.config.teacher)
            
            print("📚 Loading Dataset...")
            self.data_manager = DataManager(self.config.training.dataset_path)
            
            print("✅ All components loaded successfully!\n")
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            raise
    
    def test_basic_responses(self):
        """Test basic student responses"""
        print("🧪 BASIC RESPONSE TEST")
        print("=" * 50)
        
        test_questions = [
            "What is the capital of France?",
            "How many days are in a year?",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nTest {i}/5:")
            print(f"Question: {question}")
            
            response = self.student.generate_response(question)
            print(f"Student Answer: {response}")
            
            # Get teacher evaluation
            if self.data_manager.qa_pairs:
                # Try to find correct answer from dataset
                correct_answer = None
                for qa_pair in self.data_manager.qa_pairs:
                    if question.lower() in qa_pair.question.lower():
                        correct_answer = qa_pair.answer
                        break
                
                if correct_answer:
                    evaluation = self.teacher.evaluate_answer(question, correct_answer, response)
                    print(f"Teacher Score: {evaluation['score']}/10")
                    