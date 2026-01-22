"""
Unit tests for Disaster Response Intelligence System
"""

import sys
import unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from components.embeddings.multimodal_encoders import HybridMultimodalEmbedder
from components.retrieval.qdrant_integration import QdrantVectorMemory
from components.retrieval.search_engine import RetrievalEngine
from components.agents.reasoning_agent import ReasoningAgent
from components.memory_system.evolving_memory import EvolvingMemoryBank, InteractionSession
from components.data.ingestion import DisasterDataGenerator, DataIngestionPipeline


class TestMultimodalEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embedder = HybridMultimodalEmbedder(embedding_dim=512)
    
    def test_text_embedding_shape(self):
        embedding = self.embedder.embed_text("earthquake damage")
        self.assertEqual(embedding.shape, (512,))
        self.assertEqual(embedding.dtype, np.float32)
    
    def test_embedding_normalization(self):
        embedding = self.embedder.embed_text("test query")
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_multimodal_fusion(self):
        embedding = self.embedder.embed_multimodal(text="earthquake")
        self.assertEqual(embedding.shape, (512,))
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestQdrantVectorMemory(unittest.TestCase):
    def setUp(self):
        self.memory = QdrantVectorMemory(
            collection_name="test_collection",
            vector_size=512,
            path=":memory:"
        )
    
    def test_store_and_retrieve(self):
        vector = np.random.randn(512).astype(np.float32)
        metadata = {"disaster_type": "earthquake", "severity": 4}
        
        point_id = self.memory.store_point(vector, metadata)
        self.assertIsNotNone(point_id)
    
    def test_search(self):
        vector = np.random.randn(512).astype(np.float32)
        metadata = {"disaster_type": "flood", "severity": 3, "location": "north"}
        
        self.memory.store_point(vector, metadata)
        results = self.memory.search(vector, limit=5)
        
        self.assertGreater(len(results), 0)
        self.assertIn("score", results[0])
        self.assertIn("id", results[0])
    
    def test_filtered_search(self):
        vector = np.random.randn(512).astype(np.float32)
        metadata1 = {"disaster_type": "earthquake", "severity": 5, "location": "west"}
        metadata2 = {"disaster_type": "flood", "severity": 2, "location": "north"}
        
        self.memory.store_point(vector, metadata1, point_id=1)
        self.memory.store_point(vector * 0.9, metadata2, point_id=2)
        
        results = self.memory.search(vector, limit=5)
        self.assertGreater(len(results), 0)
    
    def test_collection_stats(self):
        vector = np.random.randn(512).astype(np.float32)
        self.memory.store_point(vector, {"test": "metadata"})
        
        stats = self.memory.get_collection_stats()
        self.assertIn("points_count", stats)
        self.assertGreater(stats["points_count"], 0)


class TestRetrievalEngine(unittest.TestCase):
    def setUp(self):
        self.memory = QdrantVectorMemory(
            collection_name="retrieval_test",
            vector_size=512,
            path=":memory:"
        )
        self.embedder = HybridMultimodalEmbedder(embedding_dim=512)
        self.engine = RetrievalEngine(self.memory, self.embedder)
        
        for i in range(10):
            vector = np.random.randn(512).astype(np.float32)
            metadata = {
                "disaster_type": "earthquake" if i % 2 == 0 else "flood",
                "severity": (i % 5) + 1,
                "location": "zone_a",
                "source_type": "report",
            }
            self.memory.store_point(vector, metadata, point_id=i)
    
    def test_semantic_search(self):
        results = self.engine.semantic_search("earthquake damage", limit=5)
        self.assertGreater(len(results), 0)
    
    def test_re_ranking(self):
        results = self.engine.semantic_search("test query", limit=5)
        reranked = self.engine.re_rank_results(
            results,
            ranking_criteria={"severity": 0.3, "recency": 0.1}
        )
        
        self.assertEqual(len(reranked), len(results))
        for result in reranked:
            self.assertIn("final_score", result)


class TestEvolvingMemory(unittest.TestCase):
    def setUp(self):
        self.memory_bank = EvolvingMemoryBank(decay_factor=0.95)
    
    def test_store_memory(self):
        vector = np.random.randn(512).astype(np.float32)
        metadata = {"content": "test event", "disaster_type": "earthquake"}
        
        self.memory_bank.store_memory("mem_1", vector, metadata)
        
        self.assertIn("mem_1", self.memory_bank.memory_records)
        self.assertEqual(self.memory_bank.memory_records["mem_1"]["relevance_score"], 1.0)
    
    def test_access_memory(self):
        vector = np.random.randn(512).astype(np.float32)
        self.memory_bank.store_memory("mem_1", vector, {})
        
        accessed = self.memory_bank.access_memory("mem_1")
        self.assertIsNotNone(accessed)
        self.assertGreater(self.memory_bank.memory_records["mem_1"]["access_count"], 0)
    
    def test_session_creation(self):
        session = self.memory_bank.create_session("sess_1", "user_1")
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, "sess_1")
        self.assertEqual(session.user_id, "user_1")
    
    def test_memory_stats(self):
        vector = np.random.randn(512).astype(np.float32)
        self.memory_bank.store_memory("mem_1", vector, {})
        
        stats = self.memory_bank.get_memory_evolution_stats()
        self.assertIn("total_memories", stats)
        self.assertEqual(stats["total_memories"], 1)


class TestReasoningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReasoningAgent()
    
    def test_reasoning_fallback(self):
        retrieved = [
            {"id": 1, "score": 0.9, "metadata": {"content": "earthquake"}},
            {"id": 2, "score": 0.8, "metadata": {"content": "aftershock"}},
        ]
        
        result = self.agent.reason_over_retrieval("what happened?", retrieved)
        
        self.assertIn("analysis", result)
        self.assertGreater(len(result["analysis"]), 0)
    
    def test_reasoning_history(self):
        retrieved = [{"id": 1, "score": 0.9, "metadata": {}}]
        
        self.agent.reason_over_retrieval("query 1", retrieved)
        self.agent.reason_over_retrieval("query 2", retrieved)
        
        history = self.agent.get_reasoning_history(limit=5)
        self.assertGreaterEqual(len(history), 2)


class TestDisasterDataGenerator(unittest.TestCase):
    def test_generate_text_report(self):
        report = DisasterDataGenerator.generate_text_report("earthquake", "downtown", 4)
        
        self.assertEqual(report["disaster_type"], "earthquake")
        self.assertEqual(report["location"], "downtown")
        self.assertEqual(report["severity"], 4)
        self.assertIn("content", report)
    
    def test_generate_dataset(self):
        dataset = DisasterDataGenerator.generate_dataset(num_events=5)
        
        self.assertEqual(len(dataset), 5)
        for event in dataset:
            self.assertIn("disaster_type", event)
            self.assertIn("modalities", event)
    
    def test_multimodal_event(self):
        event = DisasterDataGenerator.generate_multimodal_event(
            "flood", "north", 3
        )
        
        self.assertIn("text_report", event)
        self.assertIn("image_metadata", event)
        self.assertIn("audio_log", event)
        self.assertEqual(len(event["modalities"]), 3)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.embedder = HybridMultimodalEmbedder(embedding_dim=512)
        self.memory = QdrantVectorMemory(
            collection_name="integration_test",
            vector_size=512,
            path=":memory:"
        )
        self.engine = RetrievalEngine(self.memory, self.embedder)
        self.memory_bank = EvolvingMemoryBank()
        self.agent = ReasoningAgent()
    
    def test_end_to_end_workflow(self):
        dataset = DisasterDataGenerator.generate_dataset(num_events=5)
        
        pipeline = DataIngestionPipeline(self.memory, self.embedder, self.memory_bank)
        point_ids = pipeline.batch_ingest(dataset)
        
        self.assertEqual(len(point_ids), 5)
        
        results = self.engine.semantic_search("earthquake damage", limit=3)
        self.assertGreater(len(results), 0)
        
        analysis = self.agent.reason_over_retrieval(
            "what happened?", results, include_trace=True
        )
        self.assertIn("analysis", analysis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
