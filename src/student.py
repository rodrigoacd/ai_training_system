"""
Student AI Model - Small model for learning
"""
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig, pipeline
)
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class StudentAI:
    """Student AI model using small transformer models"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.device = self._setup_device()
        self.conversation_history = []
        
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup computing device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self):
        """Load the student model and tokenizer"""
        try:
            print(f"Loading student model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                padding_side='left'
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            print(f"Student model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading student model: {e}")
            # Fallback to a simpler model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if main model fails"""
        try:
            print("Loading fallback model: distilgpt2")
            fallback_model = "distilgpt2"
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            self.model = self.model.to(self.device)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            print("Fallback model loaded successfully")
            
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            raise Exception("Failed to load any student model")
    
    def generate_response(self, question: str, context: str = None) -> str:
        """Generate response to a question"""
        try:
            # Prepare the prompt with better context integration
            if context and context.strip():
                # Add context BEFORE the question for better influence
                prompt = f"{context}\n\nNow answer this question:\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            # Generate response with adjusted parameters for context
            generation_config = GenerationConfig(
                max_new_tokens=min(250, self.config.max_length),  # Increased for context
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                top_p=0.9  # Add nucleus sampling for better quality
            )
            
            outputs = self.generator(
                prompt,
                generation_config=generation_config,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            
            response = outputs[0]['generated_text'].strip()
            
            # Clean up the response
            response = self._clean_response(response, question)
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "answer": response,
                "context": context,
                "context_used": bool(context and context.strip())
            })
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I couldn't generate a proper response. Error: {str(e)}"
    
    def _clean_response(self, response: str, question: str) -> str:
        """Clean and format the generated response"""
        # Remove the original question if it appears in response
        if question in response:
            response = response.replace(question, "").strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Answer:", "A:", "Response:", "Reply:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Take only the first sentence or paragraph for clarity
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[0]) > 10:
            response = sentences[0] + '.'
        
        # Limit length
        if len(response) > 300:
            response = response[:300] + "..."
        
        return response.strip()
    
    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "temperature": self.config.temperature,
            "conversation_length": len(self.conversation_history)
        }
        
        if self.model:
            try:
                info["model_parameters"] = sum(p.numel() for p in self.model.parameters())
                info["model_size_mb"] = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
            except:
                info["model_parameters"] = "Unknown"
                info["model_size_mb"] = "Unknown"
        
        return info
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("=== Student AI Interactive Chat ===")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'clear' to clear conversation history")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'clear':
                    self.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                if not question:
                    continue
                
                print("Student AI: ", end="")
                response = self.generate_response(question)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def evaluate_on_dataset(self, qa_pairs: list) -> Dict[str, Any]:
        """Evaluate model on a set of Q&A pairs"""
        results = []
        
        for qa_pair in qa_pairs:
            try:
                response = self.generate_response(qa_pair.question)
                results.append({
                    "question": qa_pair.question,
                    "expected_answer": qa_pair.answer,
                    "student_answer": response,
                    "question_length": len(qa_pair.question),
                    "answer_length": len(response)
                })
            except Exception as e:
                results.append({
                    "question": qa_pair.question,
                    "expected_answer": qa_pair.answer,
                    "student_answer": f"Error: {e}",
                    "question_length": len(qa_pair.question),
                    "answer_length": 0
                })
        
        # Calculate basic statistics
        total_questions = len(results)
        avg_answer_length = sum(r["answer_length"] for r in results) / total_questions if total_questions > 0 else 0
        
        evaluation_results = {
            "total_questions": total_questions,
            "average_answer_length": avg_answer_length,
            "results": results,
            "model_info": self.get_model_info()
        }
        
        return evaluation_results