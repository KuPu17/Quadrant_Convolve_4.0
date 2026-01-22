import sys
import json
from datetime import datetime
from typing import Optional
import uuid


class DisasterResponseCLI:
    def __init__(self, system):
        self.system = system
        self.current_session = None
        self.session_id = str(uuid.uuid4())
    
    def start_interactive_session(self, user_id: str = "operator"):
        self.current_session = self.system.memory_bank.create_session(
            self.session_id, user_id
        )
        self._display_welcome_banner()
    
    def _display_welcome_banner(self):
        banner = """
╔═════════════════════════════════════════════════════════════╗
║   DISASTER RESPONSE INTELLIGENCE SYSTEM (DRIS) v1.0         ║
║   Multimodal Vector-Based Emergency Data Retrieval          ║
╚═════════════════════════════════════════════════════════════╝

Commands:
  search <query>           - Semantic search
  multimodal <query>       - Cross-modal search
  temporal <query>         - Time-filtered search
  context <query>          - Context-aware search
  similar <id>             - Find similar events
  analyze <query>          - Multi-hop reasoning analysis
  stats                    - System statistics
  session                  - Session information
  help                     - Display help
  exit                     - Exit system

Type 'help' for detailed command descriptions.
"""
        print(banner)
    
    def process_command(self, user_input: str) -> bool:
        if not user_input.strip():
            return True
        
        tokens = user_input.strip().split(maxsplit=1)
        command = tokens[0].lower()
        args = tokens[1] if len(tokens) > 1 else ""
        
        if command == "exit" or command == "quit":
            return False
        elif command == "search":
            self._handle_search(args)
        elif command == "multimodal":
            self._handle_multimodal(args)
        elif command == "temporal":
            self._handle_temporal(args)
        elif command == "context":
            self._handle_context(args)
        elif command == "similar":
            self._handle_similar(args)
        elif command == "analyze":
            self._handle_analyze(args)
        elif command == "stats":
            self._handle_stats()
        elif command == "session":
            self._handle_session()
        elif command == "help":
            self._display_help()
        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")
        
        return True
    
    def _handle_search(self, query: str):
        if not query:
            print("Error: Please provide a search query.")
            return
        
        print(f"\n[Semantic Search] Query: {query}")
        results = self.system.search_engine.semantic_search(query, limit=5)
        
        self._display_results(results)
        
        if self.current_session and results:
            analysis = self.system.reasoning_agent.reason_over_retrieval(query, results)
            print("\n[Analysis]")
            print(analysis["analysis"][:500] + "...\n")
            self.current_session.add_interaction(query, results, analysis["analysis"])
    
    def _handle_multimodal(self, query: str):
        if not query:
            print("Error: Please provide a search query.")
            return
        
        print(f"\n[Multimodal Search] Query: {query}")
        results = self.system.search_engine.multimodal_search(text_query=query, limit=5)
        
        self._display_results(results)
        
        if self.current_session and results:
            self.current_session.add_interaction(query, results, "Multimodal analysis complete")
    
    def _handle_temporal(self, query: str):
        if not query:
            print("Error: Please provide a search query.")
            return
        
        print(f"\n[Temporal Search] Query: {query} (Last 7 days)")
        results = self.system.search_engine.temporal_search(query, limit=5)
        
        self._display_results(results)
    
    def _handle_context(self, query: str):
        if not query:
            print("Error: Please provide a search query.")
            return
        
        print(f"\n[Context-Aware Search] Query: {query}")
        session_context = self.current_session.context if self.current_session else {}
        results = self.system.search_engine.contextual_search(query, session_context, limit=5)
        
        self._display_results(results)
    
    def _handle_similar(self, point_id_str: str):
        try:
            point_id = int(point_id_str)
            print(f"\n[Similarity Search] Finding similar events to ID: {point_id}")
            results = self.system.search_engine.similarity_search_by_id(point_id, limit=5)
            
            self._display_results(results)
        except ValueError:
            print("Error: Invalid point ID. Please provide a valid integer.")
    
    def _handle_analyze(self, query: str):
        if not query:
            print("Error: Please provide a query for analysis.")
            return
        
        print(f"\n[Multi-Hop Analysis] Query: {query}")
        analysis = self.system.reasoning_agent.multi_hop_reasoning(
            query, self.system.search_engine, max_hops=2
        )
        
        print(f"Initial Query: {analysis['initial_query']}")
        print(f"Total Sources Gathered: {analysis['total_results_gathered']}\n")
        
        for chain in analysis["reasoning_chain"]:
            print(f"  Hop {chain['hop']}: {chain['query']}")
            print(f"    - Sources Found: {chain['results_count']}\n")
        
        print("Synthesis:")
        print(analysis["final_synthesis"])
    
    def _handle_stats(self):
        stats = self.system.qdrant_memory.get_collection_stats()
        memory_stats = self.system.memory_bank.get_memory_evolution_stats()
        
        print("\n[System Statistics]")
        print(f"Vector Points Stored: {stats.get('points_count', 0)}")
        print(f"Total Memories: {memory_stats.get('total_memories', 0)}")
        print(f"Active Sessions: {memory_stats.get('active_sessions', 0)}")
        print(f"Average Memory Relevance: {memory_stats.get('average_relevance', 0):.3f}")
        print(f"Memory Types: {memory_stats.get('memory_types_distribution', {})}")
    
    def _handle_session(self):
        if not self.current_session:
            print("No active session.")
            return
        
        summary = self.current_session.get_session_summary()
        print("\n[Session Information]")
        print(f"Session ID: {summary['session_id']}")
        print(f"User ID: {summary['user_id']}")
        print(f"Interactions: {summary['interaction_count']}")
        print(f"Last Activity: {summary['last_activity']}")
    
    def _display_results(self, results):
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. [ID: {result.get('id')}] Score: {result.get('score', 0):.3f}")
            payload = result.get("payload", {})
            print(f"   Type: {payload.get('source_type', 'unknown')}")
            print(f"   Disaster: {payload.get('disaster_type', 'unknown')}")
            print(f"   Location: {payload.get('location', 'unknown')}")
            print(f"   Severity: {payload.get('severity', 0)}/5")
            print()
    
    def _display_help(self):
        help_text = """
DISASTER RESPONSE INTELLIGENCE SYSTEM - HELP

Commands:

1. search <query>
   Performs semantic similarity search across all stored data.
   Example: search "earthquake damage assessment"

2. multimodal <query>
   Cross-modal search across text, images, and audio.
   Example: multimodal "coastal flooding"

3. temporal <query>
   Search limited to recent events (last 7 days).
   Example: temporal "active fire spread"

4. context <query>
   Uses session context for intelligent filtering.
   Example: context "emergency response"

5. similar <id>
   Finds events similar to a specific point ID.
   Example: similar 12345

6. analyze <query>
   Performs multi-hop reasoning for deep analysis.
   Example: analyze "cascade effects of earthquake"

7. stats
   Displays system statistics and memory evolution.
   Example: stats

8. session
   Shows current session information.
   Example: session

9. help
   Shows this help text.
   Example: help

10. exit/quit
    Exit the system.
    Example: exit
"""
        print(help_text)


def run_cli_interface(system):
    cli = DisasterResponseCLI(system)
    cli.start_interactive_session("disaster_operator")
    
    print("\nEnter 'help' for command list or 'exit' to quit.\n")
    
    try:
        while True:
            user_input = input("DRIS> ").strip()
            if not cli.process_command(user_input):
                print("\nExiting Disaster Response Intelligence System. Goodbye!")
                break
    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
    except Exception as e:
        print(f"Error: {e}")
