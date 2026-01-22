#!/usr/bin/env python
"""
Disaster Response Intelligence System - Main Entry Point
Comprehensive multimodal vector-based retrieval and memory system for emergency response
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from embeddings.multimodal_encoders import HybridMultimodalEmbedder
from retrieval.qdrant_integration import QdrantVectorMemory
from retrieval.search_engine import RetrievalEngine
from agents.reasoning_agent import ReasoningAgent
from memory_system.evolving_memory import EvolvingMemoryBank
from data.ingestion import DisasterDataGenerator, DataIngestionPipeline
from utils.cli_interface import run_cli_interface


class DisasterResponseIntelligenceSystem:
    def __init__(self, embedding_dim=512, db_path="./qdrant_storage", collection_name="disaster_response_memory"):
        print("Initializing Disaster Response Intelligence System...")
        
        self.embedder = HybridMultimodalEmbedder(embedding_dim=embedding_dim)
        print("  Multimodal embedder initialized")
        
        self.qdrant_memory = QdrantVectorMemory(
            collection_name=collection_name,
            vector_size=embedding_dim,
            path=db_path if db_path != ":memory:" else db_path
        )
        print("  Qdrant vector database initialized")
        
        self.memory_bank = EvolvingMemoryBank(decay_factor=0.95)
        print("  Evolving memory bank initialized")
        
        self.search_engine = RetrievalEngine(self.qdrant_memory, self.embedder)
        print("  Retrieval engine initialized")
        
        self.reasoning_agent = ReasoningAgent(llm_model=None)
        print("  Reasoning agent initialized")
        
        self.ingestion_pipeline = DataIngestionPipeline(
            self.qdrant_memory,
            self.embedder,
            self.memory_bank
        )
        print("  Data ingestion pipeline initialized")
    
    def initialize_with_demo_data(self, num_events=25):
        print(f"\nGenerating {num_events} synthetic disaster events...")
        dataset = DisasterDataGenerator.generate_dataset(num_events)
        
        print(f"Ingesting data into vector database...")
        point_ids = self.ingestion_pipeline.batch_ingest(dataset)
        total_points = sum(len(ids) for ids in point_ids)
        
        print(f"  Stored {total_points} vector points")
        return total_points
    
    def demo_search(self):
        print("\n" + "="*60)
        print("DEMONSTRATION: Semantic Search")
        print("="*60)
        
        queries = [
            "earthquake structural damage",
            "flood water displacement",
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = self.search_engine.semantic_search(query, limit=2)
            for result in results:
                print(f"  Score: {result['score']:.3f} - "
                      f"{result['payload'].get('disaster_type')} at "
                      f"{result['payload'].get('location')}")
    
    def demo_reasoning(self):
        print("\n" + "="*60)
        print("DEMONSTRATION: Multi-Hop Reasoning")
        print("="*60)
        
        query = "emergency response cascades"
        print(f"\nAnalyzing: {query}")
        
        analysis = self.reasoning_agent.multi_hop_reasoning(
            query, self.search_engine, max_hops=2
        )
        
        print(f"Investigation depth: {len(analysis['reasoning_chain'])} hops")
        print(f"Total sources: {analysis['total_results_gathered']}")
    
    def show_stats(self):
        print("\n" + "="*60)
        print("SYSTEM STATISTICS")
        print("="*60)
        
        stats = self.qdrant_memory.get_collection_stats()
        memory_stats = self.memory_bank.get_memory_evolution_stats()
        
        print(f"Vector Points: {stats['points_count']}")
        print(f"Total Memories: {memory_stats['total_memories']}")
        print(f"Average Relevance: {memory_stats['average_relevance']:.3f}")


def main():
    system = DisasterResponseIntelligenceSystem(db_path=":memory:")
    
    print("\nInitializing system with demo data...")
    system.initialize_with_demo_data(num_events=20)
    
    system.demo_search()
    system.demo_reasoning()
    system.show_stats()
    
    print("\n" + "="*60)
    print("LAUNCHING INTERACTIVE CLI")
    print("="*60)
    print("\nType 'help' for available commands or 'exit' to quit.\n")
    
    run_cli_interface(system)


if __name__ == "__main__":
    main()
