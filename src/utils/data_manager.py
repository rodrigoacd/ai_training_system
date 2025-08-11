"""
Data Manager for handling Q&A datasets
"""
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class QAPair:
    """Data structure for Question-Answer pairs"""
    question: str
    answer: str
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DataManager:
    """Manages Q&A dataset operations"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.qa_pairs: List[QAPair] = []
        self.current_index = 0
        
        if self.dataset_path.exists():
            self.load_dataset()
    
    def load_dataset(self) -> bool:
        """Load Q&A pairs from CSV file"""
        try:
            df = pd.read_csv(self.dataset_path)
            
            # Validate required columns
            if 'question' not in df.columns or 'answer' not in df.columns:
                raise ValueError("CSV must contain 'question' and 'answer' columns")
            
            self.qa_pairs = []
            for _, row in df.iterrows():
                # Handle potential metadata columns
                metadata = {}
                for col in df.columns:
                    if col not in ['question', 'answer']:
                        metadata[col] = row[col]
                
                qa_pair = QAPair(
                    question=str(row['question']).strip(),
                    answer=str(row['answer']).strip(),
                    metadata=metadata if metadata else None
                )
                self.qa_pairs.append(qa_pair)
            
            print(f"Loaded {len(self.qa_pairs)} Q&A pairs from {self.dataset_path}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def get_random_question(self) -> Optional[QAPair]:
        """Get a random Q&A pair"""
        if not self.qa_pairs:
            return None
        
        return random.choice(self.qa_pairs)
    
    def get_next_question(self) -> Optional[QAPair]:
        """Get next Q&A pair in sequence"""
        if not self.qa_pairs:
            return None
        
        if self.current_index >= len(self.qa_pairs):
            self.current_index = 0
        
        qa_pair = self.qa_pairs[self.current_index]
        self.current_index += 1
        
        return qa_pair
    
    def get_batch_questions(self, batch_size: int) -> List[QAPair]:
        """Get a batch of Q&A pairs"""
        batch = []
        for _ in range(batch_size):
            qa_pair = self.get_next_question()
            if qa_pair:
                batch.append(qa_pair)
            else:
                break
        
        return batch
    
    def get_questions_by_criteria(self, criteria: Dict) -> List[QAPair]:
        """Get questions matching specific criteria from metadata"""
        matching_pairs = []
        
        for qa_pair in self.qa_pairs:
            if not qa_pair.metadata:
                continue
                
            match = True
            for key, value in criteria.items():
                if key not in qa_pair.metadata or qa_pair.metadata[key] != value:
                    match = False
                    break
            
            if match:
                matching_pairs.append(qa_pair)
        
        return matching_pairs
    
    def shuffle_dataset(self):
        """Shuffle the dataset"""
        random.shuffle(self.qa_pairs)
        self.current_index = 0
    
    def split_dataset(self, train_ratio: float = 0.8) -> Tuple[List[QAPair], List[QAPair]]:
        """Split dataset into training and testing sets"""
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        
        shuffled_pairs = self.qa_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        split_index = int(len(shuffled_pairs) * train_ratio)
        train_set = shuffled_pairs[:split_index]
        test_set = shuffled_pairs[split_index:]
        
        return train_set, test_set
    
    def create_test_file(self, output_path: str, test_size: int = 5):
        """Create a separate test file with random questions"""
        if len(self.qa_pairs) < test_size:
            test_size = len(self.qa_pairs)
        
        test_pairs = random.sample(self.qa_pairs, test_size)
        
        # Convert to DataFrame and save
        test_data = {
            'question': [pair.question for pair in test_pairs],
            'answer': [pair.answer for pair in test_pairs]
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv(output_path, index=False)
        
        print(f"Created test file with {len(test_pairs)} questions at {output_path}")
        return output_path
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        if not self.qa_pairs:
            return {"total_questions": 0}
        
        info = {
            "total_questions": len(self.qa_pairs),
            "current_index": self.current_index,
            "dataset_path": str(self.dataset_path),
            "sample_question": self.qa_pairs[0].question if self.qa_pairs else None,
            "metadata_fields": list(self.qa_pairs[0].metadata.keys()) if self.qa_pairs and self.qa_pairs[0].metadata else []
        }
        
        return info
    
    def add_qa_pair(self, question: str, answer: str, metadata: Dict = None):
        """Add a new Q&A pair to the dataset"""
        qa_pair = QAPair(question=question, answer=answer, metadata=metadata)
        self.qa_pairs.append(qa_pair)
    
    def save_dataset(self, output_path: str):
        """Save current dataset to CSV file"""
        if not self.qa_pairs:
            print("No data to save")
            return False
        
        data = {
            'question': [pair.question for pair in self.qa_pairs],
            'answer': [pair.answer for pair in self.qa_pairs]
        }
        
        # Add metadata columns if they exist
        if self.qa_pairs[0].metadata:
            for key in self.qa_pairs[0].metadata.keys():
                data[key] = [pair.metadata.get(key, '') for pair in self.qa_pairs]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(self.qa_pairs)} Q&A pairs to {output_path}")
        return True