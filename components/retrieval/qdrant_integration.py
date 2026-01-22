from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, HasIdCondition
from qdrant_client.models import Range, MatchAny
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import json
import uuid


class QdrantVectorMemory:
    def __init__(self, 
                 collection_name: str = "disaster_response_memory",
                 vector_size: int = 512,
                 host: str = "localhost",
                 port: int = 6333,
                 path: Optional[str] = None):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        if path:
            if path == ":memory:":
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(path=path)
        else:
            try:
                self.client = QdrantClient(host=host, port=port)
            except:
                self.client = QdrantClient(":memory:")
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
    
    def store_point(self, 
                   vector: np.ndarray,
                   metadata: Dict[str, Any],
                   point_id: Optional[int] = None) -> int:
        if point_id is None:
            point_id = abs(hash(str(metadata))) % (10 ** 8)
        
        payload = {
            "metadata": json.dumps(metadata),
            "timestamp": datetime.utcnow().isoformat(),
            "modalities": metadata.get("modalities", []),
            "disaster_type": metadata.get("disaster_type", "unknown"),
            "severity": metadata.get("severity", 0),
            "location": metadata.get("location", ""),
            "source_type": metadata.get("source_type", ""),
        }
        
        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        
        return point_id
    
    def search(self, 
              query_vector: np.ndarray,
              limit: int = 10,
              score_threshold: Optional[float] = None,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        
        filter_obj = None
        if filters:
            conditions = []
            
            if "disaster_type" in filters:
                conditions.append(
                    FieldCondition(
                        key="disaster_type",
                        match=MatchValue(value=filters["disaster_type"])
                    )
                )
            
            if "min_severity" in filters:
                conditions.append(
                    FieldCondition(
                        key="severity",
                        range=Range(gte=filters["min_severity"])
                    )
                )
            
            if "location" in filters:
                conditions.append(
                    FieldCondition(
                        key="location",
                        match=MatchValue(value=filters["location"])
                    )
                )
            
            if conditions:
                from qdrant_client.models import HasIdCondition
                if len(conditions) == 1:
                    filter_obj = conditions[0]
                else:
                    from qdrant_client.models import AND
                    filter_obj = AND(conditions=conditions)
        
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=limit,
                query_filter=filter_obj,
                score_threshold=score_threshold,
            )
            results = search_result.points
        except (AttributeError, TypeError):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=filter_obj,
                score_threshold=score_threshold,
            )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": json.loads(hit.payload.get("metadata", "{}")),
                "payload": hit.payload,
                "vector": hit.vector if hasattr(hit, 'vector') else None,
            }
            for hit in results
        ]
    
    def update_point(self, point_id: int, updates: Dict[str, Any]):
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
        )
        
        if results:
            point = results[0]
            payload = dict(point.payload)
            
            metadata = json.loads(payload.get("metadata", "{}"))
            metadata.update(updates.get("metadata", {}))
            payload["metadata"] = json.dumps(metadata)
            
            for k, v in updates.items():
                if k != "metadata":
                    payload[k] = v
            
            payload["updated_at"] = datetime.utcnow().isoformat()
            
            point_struct = PointStruct(
                id=point_id,
                vector=point.vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point_struct],
            )
    
    def delete_point(self, point_id: int):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
    
    def batch_search(self, 
                    query_vectors: List[np.ndarray],
                    limit: int = 10) -> List[List[Dict[str, Any]]]:
        return [self.search(qv, limit) for qv in query_vectors]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": getattr(info, 'vectors_count', info.points_count),
            "indexed_vectors_count": getattr(info, 'indexed_vectors_count', info.points_count),
            "segment_count": getattr(info, 'segments_count', 1),
        }
    
    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
    
    def rebuild_index(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
