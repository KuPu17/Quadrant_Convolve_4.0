# Disaster Response Intelligence System

A sophisticated multimodal vector-based retrieval and memory system for emergency disaster response coordination.

## Quick Start

```bash
cd DisasterResponseIntelligenceSystem
pip install -r requirements.txt
python src/main.py
```

## Features

- **Multimodal Retrieval**: Search across text reports, satellite imagery, and emergency audio
- **Qdrant-Powered Search**: Sub-millisecond vector similarity search with semantic filtering
- **Memory Evolution**: Time-decaying memory system with reinforcement learning
- **Session Management**: Context-aware queries with interaction history tracking
- **Multi-Hop Reasoning**: Chained retrieval queries for deep analysis
- **Re-Ranking Engine**: Custom criteria-based result ordering (severity, recency, confidence)
- **Interactive CLI**: Real-time query interface for emergency responders

## System Architecture

```
Multimodal Data → Embedding Generation → Qdrant Vector Database
                                              ↓
                        Retrieval Engine ← Memory Evolution System
                              ↓
                    Reasoning Agent (Grounded Analysis)
                              ↓
                      Interactive CLI Interface
```

## Core Components

### 1. Multimodal Embeddings (`components/embeddings/`)
- CLIP-based text and image encoding
- Librosa-based audio feature extraction
- Hybrid fusion for cross-modal alignment

### 2. Qdrant Integration (`components/retrieval/qdrant_integration.py`)
- Vector storage and indexing
- Metadata-based filtering (disaster_type, severity, location)
- Batch search operations

### 3. Retrieval Engine (`components/retrieval/search_engine.py`)
- Semantic search
- Multimodal search
- Temporal filtering
- Context-aware queries
- Cross-modal bridging
- Re-ranking with custom criteria

### 4. Memory System (`components/memory_system/evolving_memory.py`)
- Session-based interaction tracking
- Time-decay relevance scoring
- Memory reinforcement on access
- Automatic pruning of low-relevance memories

### 5. Reasoning Agent (`components/agents/reasoning_agent.py`)
- Grounded analysis over retrieved data
- Multi-hop reasoning chains
- Evidence attribution and confidence estimation

### 6. Data Pipeline (`components/data/ingestion.py`)
- Synthetic disaster event generation
- Multimodal document ingestion
- Batch processing

### 7. Interactive CLI (`components/utils/cli_interface.py`)
- Real-time query interface
- Session management
- System statistics

## Configuration

Edit `global_setting/settings.py` to customize:
- Vector embedding dimension (default: 512)
- Memory decay factor (default: 0.95)
- Qdrant collection name and storage path
- Disaster types and severity levels

## Example Queries

```
DRIS> search "earthquake damage assessment"
DRIS> multimodal "flood impact satellite imagery"
DRIS> temporal "emergency response in last 7 days"
DRIS> context "hurricane preparedness"
DRIS> similar 12345
DRIS> analyze "cascade effects of earthquake"
DRIS> stats
DRIS> exit
```

## License

Open source - Emergency Response Systems
