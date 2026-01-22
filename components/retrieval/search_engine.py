import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json


class RetrievalEngine:
    def __init__(self, qdrant_memory, embedder):
        self.qdrant_memory = qdrant_memory
        self.embedder = embedder
    
    def multimodal_search(self,
                         text_query: Optional[str] = None,
                         image_path: Optional[str] = None,
                         audio_path: Optional[str] = None,
                         limit: int = 10,
                         filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        
        query_vector = self.embedder.embed_multimodal(
            text=text_query,
            image_path=image_path,
            audio_path=audio_path
        )
        
        results = self.qdrant_memory.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters
        )
        
        return results
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        vector = self.embedder.embed_text(query)
        return self.qdrant_memory.search(query_vector=vector, limit=limit)
    
    def similarity_search_by_id(self, source_id: int, limit: int = 10) -> List[Dict]:
        source = self.qdrant_memory.client.retrieve(
            collection_name=self.qdrant_memory.collection_name,
            ids=[source_id],
        )
        
        if not source:
            return []
        
        vector = source[0].vector
        results = self.qdrant_memory.search(
            query_vector=np.array(vector),
            limit=limit
        )
        
        return [r for r in results if r["id"] != source_id]
    
    def contextual_search(self,
                         query: str,
                         session_context: Dict[str, Any],
                         limit: int = 10) -> List[Dict]:
        
        vector = self.embedder.embed_text(query)
        results = self.qdrant_memory.search(query_vector=vector, limit=limit)
        
        context_filters = {}
        if "disaster_type" in session_context:
            context_filters["disaster_type"] = session_context["disaster_type"]
        
        if context_filters:
            results_with_context = self.qdrant_memory.search(
                query_vector=vector,
                limit=limit,
                filters=context_filters
            )
            return results_with_context if results_with_context else results
        
        return results
    
    def cross_modal_search(self,
                          primary_modality: str,
                          query: str,
                          target_modalities: List[str],
                          limit: int = 10) -> Dict[str, List[Dict]]:
        
        if primary_modality == "text":
            vector = self.embedder.embed_text(query)
        elif primary_modality == "image":
            vector = self.embedder.embed_image(query)
        elif primary_modality == "audio":
            vector = self.embedder.embed_audio(query)
        else:
            vector = np.random.randn(512).astype(np.float32)
        
        all_results = self.qdrant_memory.search(query_vector=vector, limit=limit)
        
        organized_results = {}
        for modality in target_modalities:
            organized_results[modality] = [
                r for r in all_results
                if modality in r["payload"].get("modalities", [])
            ][:limit]
        
        organized_results["all"] = all_results
        
        return organized_results
    
    def temporal_search(self,
                       query: str,
                       time_range_days: int = 7,
                       limit: int = 10) -> List[Dict]:
        
        vector = self.embedder.embed_text(query)
        results = self.qdrant_memory.search(query_vector=vector, limit=limit*2)
        
        from datetime import datetime, timedelta
        cutoff_time = datetime.utcnow() - timedelta(days=time_range_days)
        
        filtered_results = []
        for result in results:
            payload = result["payload"]
            timestamp = payload.get("timestamp", "")
            if timestamp:
                try:
                    result_time = datetime.fromisoformat(timestamp)
                    if result_time >= cutoff_time:
                        filtered_results.append(result)
                except:
                    filtered_results.append(result)
        
        return filtered_results[:limit]
    
    def re_rank_results(self,
                       results: List[Dict],
                       ranking_criteria: Dict[str, float]) -> List[Dict]:
        
        for result in results:
            score = result.get("score", 0)
            payload = result["payload"]
            
            if "severity" in ranking_criteria:
                severity = payload.get("severity", 0)
                score += severity * ranking_criteria.get("severity", 0.1)
            
            if "recency" in ranking_criteria:
                from datetime import datetime
                timestamp = payload.get("timestamp", "")
                if timestamp:
                    try:
                        result_time = datetime.fromisoformat(timestamp)
                        age_hours = (datetime.utcnow() - result_time).total_seconds() / 3600
                        recency_boost = ranking_criteria.get("recency", 0.1) / (1 + age_hours / 24)
                        score += recency_boost
                    except:
                        pass
            
            result["final_score"] = score
        
        return sorted(results, key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)
