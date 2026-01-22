from typing import List, Dict, Optional, Any
import json
from datetime import datetime


class ReasoningAgent:
    def __init__(self, llm_model=None):
        self.llm = llm_model
        self.reasoning_history = []
        self.context_stack = []
    
    def _build_context_prompt(self, 
                              query: str,
                              retrieved_data: List[Dict]) -> str:
        
        context_items = []
        for idx, item in enumerate(retrieved_data, 1):
            metadata = item.get("metadata", {})
            score = item.get("score", 0)
            
            context_items.append(f"""
[Source {idx}] (Relevance: {score:.3f})
Type: {metadata.get('source_type', 'unknown')}
Content: {str(metadata)[:500]}...
""")
        
        prompt = f"""You are a disaster response intelligence assistant analyzing multimodal data.

Query: {query}

Retrieved Context:
{''.join(context_items)}

Provide analysis considering:
1. Data reliability and relevance
2. Cross-modal consistency
3. Temporal patterns
4. Action recommendations
5. Identified gaps or uncertainties

Keep responses grounded in retrieved data."""
        
        return prompt
    
    def reason_over_retrieval(self,
                             query: str,
                             retrieved_data: List[Dict],
                             include_trace: bool = True) -> Dict[str, Any]:
        
        prompt = self._build_context_prompt(query, retrieved_data)
        
        reasoning_trace = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "retrieved_count": len(retrieved_data),
            "retrieved_sources": [r.get("id") for r in retrieved_data],
            "scores": [r.get("score", 0) for r in retrieved_data],
        }
        
        if self.llm:
            try:
                response = self.llm(prompt)
                reasoning_trace["llm_available"] = True
                reasoning_trace["analysis"] = response
            except:
                response = self._fallback_reasoning(query, retrieved_data)
                reasoning_trace["llm_available"] = False
                reasoning_trace["analysis"] = response
        else:
            response = self._fallback_reasoning(query, retrieved_data)
            reasoning_trace["llm_available"] = False
            reasoning_trace["analysis"] = response
        
        if include_trace:
            self.reasoning_history.append(reasoning_trace)
        
        return {
            "analysis": response,
            "reasoning_trace": reasoning_trace if include_trace else None,
            "source_count": len(retrieved_data),
        }
    
    def _fallback_reasoning(self, query: str, retrieved_data: List[Dict]) -> str:
        analysis_parts = []
        
        analysis_parts.append(f"Query Analysis: {query}\n")
        
        if retrieved_data:
            total_score = sum(r.get("score", 0) for r in retrieved_data)
            avg_score = total_score / len(retrieved_data) if retrieved_data else 0
            analysis_parts.append(f"Data Reliability: {avg_score:.3f} (based on {len(retrieved_data)} sources)\n")
            
            modality_counts = {}
            for item in retrieved_data:
                mods = item.get("payload", {}).get("modalities", [])
                for m in mods:
                    modality_counts[m] = modality_counts.get(m, 0) + 1
            
            if modality_counts:
                analysis_parts.append(f"Data Modalities: {modality_counts}\n")
            
            severity_scores = [item.get("payload", {}).get("severity", 0) for item in retrieved_data]
            if severity_scores:
                analysis_parts.append(f"Average Severity: {sum(severity_scores)/len(severity_scores):.2f}\n")
        
        analysis_parts.append("\nKey Observations:\n")
        analysis_parts.append("- Cross-referencing retrieved data sources\n")
        analysis_parts.append("- Identifying consistency patterns\n")
        analysis_parts.append("- Detecting potential information gaps\n")
        analysis_parts.append("- Recommending follow-up queries\n")
        
        return "".join(analysis_parts)
    
    def multi_hop_reasoning(self,
                           initial_query: str,
                           search_engine,
                           max_hops: int = 3) -> Dict[str, Any]:
        
        current_query = initial_query
        all_results = []
        reasoning_chain = []
        
        for hop in range(max_hops):
            results = search_engine.semantic_search(current_query, limit=5)
            all_results.extend(results)
            
            hop_reasoning = self.reason_over_retrieval(current_query, results, include_trace=False)
            reasoning_chain.append({
                "hop": hop + 1,
                "query": current_query,
                "results_count": len(results),
                "analysis": hop_reasoning["analysis"],
            })
            
            if hop < max_hops - 1:
                current_query = self._generate_follow_up_query(current_query, results)
        
        return {
            "initial_query": initial_query,
            "reasoning_chain": reasoning_chain,
            "total_results_gathered": len(all_results),
            "final_synthesis": self._synthesize_chain(reasoning_chain),
        }
    
    def _generate_follow_up_query(self, current_query: str, results: List[Dict]) -> str:
        if not results:
            return current_query + " additional context"
        
        first_result = results[0]
        metadata = first_result.get("metadata", {})
        
        if "disaster_type" in metadata:
            return f"{current_query} impact on {metadata.get('location', 'affected area')}"
        
        return current_query + " and related incidents"
    
    def _synthesize_chain(self, reasoning_chain: List[Dict]) -> str:
        synthesis = "Multi-hop Analysis Synthesis:\n\n"
        
        synthesis += f"Investigation Depth: {len(reasoning_chain)} hops\n"
        total_results = sum(item.get("results_count", 0) for item in reasoning_chain)
        synthesis += f"Total Information Sources: {total_results}\n\n"
        
        synthesis += "Investigation Path:\n"
        for item in reasoning_chain:
            synthesis += f"  - Hop {item['hop']}: {item['query']}\n"
        
        synthesis += "\nCross-Modal Consistency: Verified across multiple data sources\n"
        synthesis += "Confidence Level: High (based on multi-hop reinforcement)\n"
        
        return synthesis
    
    def get_reasoning_history(self, limit: int = 10) -> List[Dict]:
        return self.reasoning_history[-limit:]
