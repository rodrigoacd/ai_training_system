"""
Teacher AI Model - Uses Groq API with Llama-3-70B
"""
import json
import time
from typing import Dict, List, Optional, Tuple
from groq import Groq
import re

class TeacherAI:
    """Teacher AI using Groq API for evaluation and guidance"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.evaluation_history = []
        self.correction_templates = self._load_correction_templates()
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client"""
        try:
            if not self.config.api_key:
                raise ValueError("Groq API key not provided")
            
            self.client = Groq(api_key=self.config.api_key)
            print("Teacher AI (Groq) initialized successfully")
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            print(f"Error initializing Teacher AI: {e}")
            raise
    
    def _test_connection(self):
        """Test Groq API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": "Hello, can you respond with 'Connection successful'?"}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            if "Connection successful" not in result:
                print("Warning: Unexpected response from Groq API")
            else:
                print("Groq API connection test passed")
                
        except Exception as e:
            print(f"Groq API connection test failed: {e}")
            raise
    
    def _load_correction_templates(self) -> Dict[str, str]:
        """Load templates for different types of corrections"""
        return {
            "evaluation": """
            You are an expert teacher evaluating a student's answer. 
            
            Question: {question}
            Correct Answer: {correct_answer}
            Student Answer: {student_answer}
            
            Please evaluate the student's answer and provide:
            1. A score from 0-10 (10 being perfect)
            2. Specific feedback on what was correct/incorrect
            3. Suggestions for improvement
            4. A corrected version if needed
            
            Format your response as JSON:
            {{
                "score": <number>,
                "feedback": "<detailed feedback>",
                "suggestions": "<improvement suggestions>",
                "corrected_answer": "<corrected version if needed>"
            }}
            """,
            
            "question_generation": """
            Generate {num_questions} educational questions about the topic: {topic}
            
            Requirements:
            - Questions should be clear and specific
            - Include a mix of difficulty levels
            - Provide the correct answer for each question
            - Make questions engaging and educational
            
            Format as JSON:
            {{
                "questions": [
                    {{"question": "<question>", "answer": "<answer>", "difficulty": "<easy/medium/hard>"}},
                    ...
                ]
            }}
            """,
            
            "feedback": """
            You are a supportive teacher providing constructive feedback.
            
            The student attempted to answer: "{question}"
            Their response was: "{student_answer}"
            
            Provide encouraging but honest feedback that helps them learn.
            Focus on what they can do better next time.
            Keep the tone positive and educational.
            """
        }
    
    def evaluate_answer(self, question: str, correct_answer: str, 
                       student_answer: str) -> Dict[str, any]:
        """Evaluate student's answer against the correct answer"""
        try:
            prompt = self.correction_templates["evaluation"].format(
                question=question,
                correct_answer=correct_answer,
                student_answer=student_answer
            )
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert teacher who provides fair, detailed evaluations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                
                # Ensure all required fields exist
                required_fields = ["score", "feedback", "suggestions", "corrected_answer"]
                for field in required_fields:
                    if field not in result:
                        result[field] = ""
                
                # Validate score
                if not isinstance(result["score"], (int, float)) or not (0 <= result["score"] <= 10):
                    result["score"] = 0
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = self._parse_text_evaluation(result_text)
            
            # Add metadata
            result["timestamp"] = time.time()
            result["question"] = question
            result["correct_answer"] = correct_answer
            result["student_answer"] = student_answer
            
            # Store in history
            self.evaluation_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                "score": 0,
                "feedback": f"Error occurred during evaluation: {str(e)}",
                "suggestions": "Please try again.",
                "corrected_answer": correct_answer,
                "timestamp": time.time(),
                "question": question,
                "correct_answer": correct_answer,
                "student_answer": student_answer
            }
    
    def _parse_text_evaluation(self, text: str) -> Dict[str, any]:
        """Parse evaluation from plain text if JSON parsing fails"""
        result = {
            "score": 0,
            "feedback": text,
            "suggestions": "",
            "corrected_answer": ""
        }
        
        # Try to extract score using regex
        score_match = re.search(r'score[:\s]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 10:
                    result["score"] = score
            except:
                pass
        
        return result
    
    def generate_questions(self, topic: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """Generate questions about a specific topic"""
        try:
            prompt = self.correction_templates["question_generation"].format(
                topic=topic,
                num_questions=num_questions
            )
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert educator who creates engaging questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.7
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
                return result.get("questions", [])
            except json.JSONDecodeError:
                # Fallback to manual parsing
                return self._parse_text_questions(result_text)
                
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def _parse_text_questions(self, text: str) -> List[Dict[str, str]]:
        """Parse questions from plain text if JSON parsing fails"""
        questions = []
        
        # Simple regex to find question-answer pairs
        patterns = [
            r'Question[:\s]*(.+?)\s*Answer[:\s]*(.+?)(?=Question|$)',
            r'Q[:\s]*(.+?)\s*A[:\s]*(.+?)(?=Q|$)',
            r'(\d+\.?\s*.+?\?)\s*(.+?)(?=\d+\.|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                for match in matches:
                    questions.append({
                        "question": match[0].strip(),
                        "answer": match[1].strip(),
                        "difficulty": "medium"
                    })
                break
        
        return questions[:5]  # Limit to 5 questions
    
    def provide_feedback(self, question: str, student_answer: str) -> str:
        """Provide general feedback without requiring correct answer"""
        try:
            prompt = self.correction_templates["feedback"].format(
                question=question,
                student_answer=student_answer
            )
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a supportive teacher providing constructive feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I'm having trouble providing feedback right now. Error: {str(e)}"
    
    def batch_evaluate(self, qa_evaluations: List[Tuple[str, str, str]]) -> List[Dict[str, any]]:
        """Evaluate multiple Q&A pairs in batch"""
        results = []
        
        for i, (question, correct_answer, student_answer) in enumerate(qa_evaluations):
            print(f"Evaluating {i+1}/{len(qa_evaluations)}: {question[:50]}...")
            
            result = self.evaluate_answer(question, correct_answer, student_answer)
            results.append(result)
            
            # Add small delay to respect rate limits
            time.sleep(0.5)
        
        return results
    
    def get_evaluation_statistics(self) -> Dict[str, any]:
        """Get statistics from evaluation history"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        scores = [eval_result["score"] for eval_result in self.evaluation_history]
        
        stats = {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": {
                "excellent (9-10)": len([s for s in scores if 9 <= s <= 10]),
                "good (7-8)": len([s for s in scores if 7 <= s < 9]),
                "fair (5-6)": len([s for s in scores if 5 <= s < 7]),
                "poor (0-4)": len([s for s in scores if 0 <= s < 5])
            }
        }
        
        return stats
    
    def clear_history(self):
        """Clear evaluation history"""
        self.evaluation_history = []
        print("Teacher evaluation history cleared")
    
    def get_recent_evaluations(self, limit: int = 5) -> List[Dict[str, any]]:
        """Get recent evaluations"""
        return self.evaluation_history[-limit:] if self.evaluation_history else []