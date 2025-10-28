"""
FAISS-based Memory Module for AI Training System
Stores and retrieves past interactions (question-answer-feedback triples)
using semantic similarity search.
"""
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict


@dataclass
class MemoryEntry:
    """Represents a single interaction stored in memory"""
    question: str
    student_answer: str
    teacher_feedback: str
    score: float
    timestamp: str
    session_id: str
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary without embedding (for JSON serialization)"""
        data = asdict(self)
        data.pop('embedding', None)
        return data
    
    def get_combined_text(self) -> str:
        """Combine all text fields for embedding generation"""
        return f"Question: {self.question}\nAnswer: {self.student_answer}\nFeedback: {self.teacher_feedback}"


class FAISSMemoryModule:
    """
    FAISS-based memory system for storing and retrieving past interactions.
    
    This module enables the Student AI to learn from previous Q&A sessions
    by retrieving semantically similar interactions when answering new questions.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "L2",
        dimension: int = 384,
        storage_path: str = "memory"
    ):
        """
        Initialize the memory module.
        
        Args:
            embedding_model: SentenceTransformer model name for embeddings
            index_type: "L2" for Euclidean distance or "cosine" for cosine similarity
            dimension: Embedding dimension (384 for MiniLM, 768 for BERT-base)
            storage_path: Directory to store FAISS index and metadata
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.encoder = SentenceTransformer(embedding_model)
        
        # Verify dimension matches model output
        test_embedding = self.encoder.encode(["test"], show_progress_bar=False)
        actual_dimension = test_embedding.shape[1]
        if actual_dimension != dimension:
            print(f"Warning: Model dimension {actual_dimension} != specified {dimension}")
            print(f"Adjusting dimension to {actual_dimension}")
            self.dimension = actual_dimension
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Memory storage
        self.memories: List[MemoryEntry] = []
        self.memory_count = 0
        
        print(f"Memory module initialized with {index_type} index (dim={self.dimension})")
    
    def _create_index(self) -> faiss.Index:
        """
        Create a FAISS index based on the specified type.
        
        Returns:
            FAISS index object
        """
        if self.index_type.lower() == "cosine":
            # Cosine similarity: normalize vectors and use L2 distance
            index = faiss.IndexFlatIP(self.dimension)  # Inner Product
        else:
            # L2 (Euclidean) distance
            index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def add_interaction(
        self,
        question: str,
        student_answer: str,
        teacher_feedback: str,
        score: float,
        session_id: str
    ) -> int:
        """
        Store a new interaction in memory.
        
        Args:
            question: The question asked
            student_answer: Student's response
            teacher_feedback: Teacher's evaluation feedback
            score: Evaluation score (0-10)
            session_id: Training session identifier
            
        Returns:
            Index of the added memory entry
        """
        # Create memory entry
        memory = MemoryEntry(
            question=question,
            student_answer=student_answer,
            teacher_feedback=teacher_feedback,
            score=score,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
        # Generate embedding from combined text
        combined_text = memory.get_combined_text()
        embedding = self.encoder.encode([combined_text], show_progress_bar=False)[0]
        
        # Normalize for cosine similarity
        if self.index_type.lower() == "cosine":
            embedding = embedding / np.linalg.norm(embedding)
        
        memory.embedding = embedding
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Store memory entry
        self.memories.append(memory)
        self.memory_count += 1
        
        return self.memory_count - 1
    
    def retrieve_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        exclude_session: Optional[str] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve top-k most similar past interactions.
        
        Args:
            query: Question or text to search for
            top_k: Number of similar memories to retrieve
            min_score: Minimum teacher score threshold (optional filter)
            exclude_session: Exclude memories from specific session (optional)
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        if self.memory_count == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.encoder.encode([query], show_progress_bar=False)[0]
        
        # Normalize for cosine similarity
        if self.index_type.lower() == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search FAISS index
        # Request more results for filtering
        search_k = min(top_k * 3, self.memory_count)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            search_k
        )
        
        # Convert distances to similarity scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            memory = self.memories[idx]
            
            # Apply filters
            if min_score is not None and memory.score < min_score:
                continue
            if exclude_session and memory.session_id == exclude_session:
                continue
            
            # Convert distance to similarity score
            # For L2: smaller distance = more similar
            # For cosine (IP): larger value = more similar
            if self.index_type.lower() == "cosine":
                similarity = float(dist)  # Already similarity (inner product)
            else:
                similarity = 1.0 / (1.0 + float(dist))  # Convert L2 to similarity
            
            results.append((memory, similarity))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def format_context(
        self,
        similar_memories: List[Tuple[MemoryEntry, float]],
        max_contexts: int = 3
    ) -> str:
        """
        Format retrieved memories as context string for prompt injection.
        
        Args:
            similar_memories: List of (MemoryEntry, similarity_score) tuples
            max_contexts: Maximum number of contexts to include
            
        Returns:
            Formatted context string
        """
        if not similar_memories:
            return ""
        
        context_parts = ["Here are some relevant past interactions to help you:\n"]
        
        for i, (memory, similarity) in enumerate(similar_memories[:max_contexts], 1):
            context_parts.append(f"\n--- Example {i} (Similarity: {similarity:.3f}, Score: {memory.score}/10) ---")
            context_parts.append(f"Previous Question: {memory.question}")
            context_parts.append(f"Previous Answer: {memory.student_answer}")
            context_parts.append(f"Teacher Feedback: {memory.teacher_feedback}")
        
        context_parts.append("\nUse these examples to improve your answer.\n")
        return "\n".join(context_parts)
    
    def save(self, filename_prefix: str = "memory") -> Tuple[str, str]:
        """
        Save FAISS index and metadata to disk for persistence.
        
        Args:
            filename_prefix: Prefix for saved files
            
        Returns:
            Tuple of (index_path, metadata_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save FAISS index
        index_path = self.storage_path / f"{filename_prefix}_index_{timestamp}.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata (memories without embeddings)
        metadata = {
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "memory_count": self.memory_count,
            "timestamp": timestamp,
            "memories": [memory.to_dict() for memory in self.memories]
        }
        
        metadata_path = self.storage_path / f"{filename_prefix}_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Memory saved: {self.memory_count} entries")
        print(f"  Index: {index_path}")
        print(f"  Metadata: {metadata_path}")
        
        return str(index_path), str(metadata_path)
    
    def load(self, index_path: str, metadata_path: str) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Verify compatibility
            if metadata["dimension"] != self.dimension:
                print(f"Warning: Dimension mismatch ({metadata['dimension']} vs {self.dimension})")
                return False
            
            # Reconstruct memory entries (without embeddings)
            self.memories = []
            for mem_data in metadata["memories"]:
                memory = MemoryEntry(**mem_data)
                self.memories.append(memory)
            
            self.memory_count = len(self.memories)
            
            print(f"Memory loaded: {self.memory_count} entries")
            print(f"  Model: {metadata['embedding_model']}")
            print(f"  Index type: {metadata['index_type']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading memory: {e}")
            return False
    
    def prune_memory(
        self,
        max_entries: Optional[int] = None,
        min_score: Optional[float] = None,
        keep_recent: int = 100
    ) -> int:
        """
        Prune memory to improve performance and remove low-quality entries.
        
        Args:
            max_entries: Maximum total entries to keep (oldest removed first)
            min_score: Remove entries below this score threshold
            keep_recent: Always keep this many most recent entries
            
        Returns:
            Number of entries removed
        """
        initial_count = self.memory_count
        
        if self.memory_count == 0:
            return 0
        
        # Filter by score
        if min_score is not None:
            # Sort by timestamp to preserve recent entries
            sorted_memories = sorted(
                enumerate(self.memories),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            
            kept_indices = []
            kept_memories = []
            
            for i, (orig_idx, memory) in enumerate(sorted_memories):
                # Keep recent entries regardless of score
                if i < keep_recent or memory.score >= min_score:
                    kept_indices.append(orig_idx)
                    kept_memories.append(memory)
            
            self.memories = kept_memories
        
        # Limit total entries
        if max_entries is not None and len(self.memories) > max_entries:
            # Keep most recent entries
            sorted_memories = sorted(self.memories, key=lambda x: x.timestamp, reverse=True)
            self.memories = sorted_memories[:max_entries]
        
        # Rebuild FAISS index
        if len(self.memories) < initial_count:
            self._rebuild_index()
        
        removed_count = initial_count - len(self.memories)
        self.memory_count = len(self.memories)
        
        print(f"Memory pruned: removed {removed_count} entries, {self.memory_count} remaining")
        return removed_count
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current memories"""
        print("Rebuilding FAISS index...")
        
        # Create new index
        self.index = self._create_index()
        
        # Re-generate embeddings if needed and add to index
        embeddings = []
        for memory in self.memories:
            if memory.embedding is None:
                combined_text = memory.get_combined_text()
                embedding = self.encoder.encode([combined_text], show_progress_bar=False)[0]
                
                if self.index_type.lower() == "cosine":
                    embedding = embedding / np.linalg.norm(embedding)
                
                memory.embedding = embedding
            
            embeddings.append(memory.embedding)
        
        # Add all embeddings to index at once
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
        
        print(f"Index rebuilt with {len(embeddings)} entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory module statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.memory_count == 0:
            return {
                "total_memories": 0,
                "index_type": self.index_type,
                "dimension": self.dimension
            }
        
        scores = [m.score for m in self.memories]
        sessions = list(set(m.session_id for m in self.memories))
        
        return {
            "total_memories": self.memory_count,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model_name,
            "average_score": np.mean(scores),
            "score_range": (np.min(scores), np.max(scores)),
            "unique_sessions": len(sessions),
            "oldest_entry": min(m.timestamp for m in self.memories),
            "newest_entry": max(m.timestamp for m in self.memories)
        }
    
    def clear_memory(self):
        """Clear all memories and reset index"""
        self.index = self._create_index()
        self.memories = []
        self.memory_count = 0
        print("Memory cleared")