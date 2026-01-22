import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from collections import deque
from enum import Enum


class MemoryType(Enum):
    IMMEDIATE = "immediate"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class InteractionSession:
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.interactions = deque(maxlen=100)
        self.context = {}
        self.memory_reinforcement = {}
    
    def add_interaction(self, query: str, results: List[Dict], response: str):
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "results_count": len(results),
            "response": response,
            "result_ids": [r.get("id") for r in results],
        }
        self.interactions.append(interaction)
        self.last_activity = datetime.utcnow()
    
    def update_context(self, key: str, value: Any):
        self.context[key] = value
    
    def reinforce_memory(self, point_id: int, reinforcement_value: float = 1.0):
        current = self.memory_reinforcement.get(point_id, 0)
        self.memory_reinforcement[point_id] = current + reinforcement_value
    
    def get_session_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "interaction_count": len(self.interactions),
            "context": self.context,
            "reinforced_memories": self.memory_reinforcement,
        }


class EvolvingMemoryBank:
    def __init__(self, decay_factor: float = 0.95, decay_period_hours: int = 24):
        self.memory_records = {}
        self.decay_factor = decay_factor
        self.decay_period = timedelta(hours=decay_period_hours)
        self.access_counts = {}
        self.sessions = {}
    
    def store_memory(self, 
                    memory_id: str,
                    vector: np.ndarray,
                    content: Dict[str, Any],
                    memory_type: MemoryType = MemoryType.EPISODIC):
        self.memory_records[memory_id] = {
            "vector": vector,
            "content": content,
            "memory_type": memory_type,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "access_count": 0,
            "relevance_score": 1.0,
            "decay_rate": 1.0,
        }
        self.access_counts[memory_id] = 0
    
    def update_memory(self, 
                     memory_id: str,
                     content_updates: Dict[str, Any]):
        if memory_id in self.memory_records:
            self.memory_records[memory_id]["content"].update(content_updates)
            self.memory_records[memory_id]["last_modified"] = datetime.utcnow().isoformat()
    
    def delete_memory(self, memory_id: str):
        if memory_id in self.memory_records:
            del self.memory_records[memory_id]
        if memory_id in self.access_counts:
            del self.access_counts[memory_id]
    
    def decay_memory(self, memory_id: str):
        if memory_id in self.memory_records:
            created = datetime.fromisoformat(self.memory_records[memory_id]["created_at"].isoformat())
            age = datetime.utcnow() - created
            decay_periods = age / self.decay_period
            
            decay_factor = self.decay_factor ** decay_periods
            self.memory_records[memory_id]["decay_rate"] = decay_factor
            self.memory_records[memory_id]["relevance_score"] *= decay_factor
    
    def access_memory(self, memory_id: str) -> Optional[Dict]:
        if memory_id in self.memory_records:
            self.memory_records[memory_id]["last_accessed"] = datetime.utcnow()
            self.memory_records[memory_id]["access_count"] += 1
            self.access_counts[memory_id] = self.memory_records[memory_id]["access_count"]
            
            self.decay_memory(memory_id)
            
            return self.memory_records[memory_id]
        return None
    
    def reinforce_memory(self, memory_id: str, reinforcement: float = 0.1):
        if memory_id in self.memory_records:
            current_relevance = self.memory_records[memory_id]["relevance_score"]
            self.memory_records[memory_id]["relevance_score"] = min(2.0, current_relevance + reinforcement)
    
    def create_session(self, session_id: str, user_id: str) -> InteractionSession:
        session = InteractionSession(session_id, user_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[InteractionSession]:
        return self.sessions.get(session_id)
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        if session_id in self.sessions:
            return self.sessions[session_id].context
        return {}
    
    def get_memory_evolution_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self.memory_records),
            "active_sessions": len(self.sessions),
            "average_relevance": np.mean([m["relevance_score"] for m in self.memory_records.values()]) if self.memory_records else 0,
            "most_accessed": max(self.access_counts.items(), key=lambda x: x[1])[0] if self.access_counts else None,
            "memory_types_distribution": self._get_memory_type_distribution(),
        }
    
    def _get_memory_type_distribution(self) -> Dict[str, int]:
        distribution = {}
        for memory in self.memory_records.values():
            mtype = memory["memory_type"].value
            distribution[mtype] = distribution.get(mtype, 0) + 1
        return distribution
    
    def prune_low_relevance_memories(self, threshold: float = 0.1):
        to_delete = [mid for mid, mem in self.memory_records.items() 
                    if mem["relevance_score"] < threshold]
        for mid in to_delete:
            self.delete_memory(mid)
        return to_delete
