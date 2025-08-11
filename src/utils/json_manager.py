"""
JSON Manager for storing training data and results
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class JSONManager:
    """Manages JSON file operations for training data"""
    
    def __init__(self, base_path: str = "logs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.corrections_path = self.base_path / "corrections"
        self.logs_path = self.base_path / "training_logs"
        self.evaluations_path = self.base_path / "evaluations"
        
        for path in [self.corrections_path, self.logs_path, self.evaluations_path]:
            path.mkdir(exist_ok=True)
    
    def save_correction(self, question: str, student_answer: str, 
                       teacher_feedback: str, score: float, 
                       session_id: str = None) -> str:
        """Save a correction from teacher to student"""
        correction_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "student_answer": student_answer,
            "teacher_feedback": teacher_feedback,
            "score": score,
            "session_id": session_id or self._generate_session_id()
        }
        
        filename = f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.corrections_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(correction_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_training_log(self, log_data: Dict[str, Any], session_id: str = None) -> str:
        """Save training session log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or self._generate_session_id(),
            **log_data
        }
        
        filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.logs_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def save_evaluation_results(self, results: Dict[str, Any], session_id: str = None) -> str:
        """Save evaluation results"""
        evaluation_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or self._generate_session_id(),
            "results": results
        }
        
        filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.evaluations_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_corrections_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all corrections for a specific session"""
        corrections = []
        
        for file_path in self.corrections_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("session_id") == session_id:
                        corrections.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading {file_path}: {e}")
        
        return sorted(corrections, key=lambda x: x["timestamp"])
    
    def load_all_corrections(self) -> List[Dict[str, Any]]:
        """Load all corrections"""
        corrections = []
        
        for file_path in self.corrections_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    corrections.append(data)
            except (json.JSONDecodeError) as e:
                print(f"Error loading {file_path}: {e}")
        
        return sorted(corrections, key=lambda x: x["timestamp"])
    
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        corrections = self.load_corrections_by_session(session_id)
        
        if not corrections:
            return {}
        
        scores = [c["score"] for c in corrections if "score" in c]
        
        stats = {
            "session_id": session_id,
            "total_corrections": len(corrections),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "start_time": corrections[0]["timestamp"],
            "end_time": corrections[-1]["timestamp"]
        }
        
        return stats
    
    def export_training_data(self, output_file: str) -> str:
        """Export all training data to a single JSON file"""
        all_data = {
            "corrections": self.load_all_corrections(),
            "export_timestamp": datetime.now().isoformat(),
            "total_corrections": len(self.load_all_corrections())
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"