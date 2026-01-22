import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class DisasterDataGenerator:
    DISASTER_TYPES = ["earthquake", "flood", "wildfire", "hurricane", "landslide", "tsunami"]
    SEVERITY_LEVELS = [1, 2, 3, 4, 5]
    MODALITIES = ["text", "image", "audio"]
    LOCATIONS = [
        "Downtown District",
        "Northern Region",
        "Eastern Valley",
        "Western Coast",
        "Central Zone",
    ]
    
    @staticmethod
    def generate_text_report(disaster_type: str, location: str, severity: int) -> Dict[str, Any]:
        reports = {
            "earthquake": f"URGENT: High magnitude earthquake detected in {location}. Severity level: {severity}/5. Multiple buildings damaged. Emergency services deployed. Aftershocks expected. Evacuation in progress.",
            "flood": f"ALERT: Major flooding event in {location}. Water levels rising rapidly. Severity: {severity}/5. Residents advised to relocate to higher ground. Roads impassable. Supply chain disrupted.",
            "wildfire": f"FIRE ALERT: Wildfire spreading rapidly in {location}. Severity: {severity}/5. Evacuation zones established. Air quality deteriorating. Power outages reported. Containment efforts ongoing.",
            "hurricane": f"STORM WARNING: Hurricane approaching {location}. Wind speeds escalating. Severity: {severity}/5. Coastal areas at risk. Shelters activated. Marine warnings issued.",
            "landslide": f"GEOLOGICAL ALERT: Landslide occurrence in {location}. Severity: {severity}/5. Infrastructure damaged. Multiple access routes blocked. Geological survey initiated.",
            "tsunami": f"TSUNAMI WARNING: Wave generated in {location}. Severity: {severity}/5. Coastal evacuations mandatory. Harbor operations suspended. International alerts issued.",
        }
        
        return {
            "type": "emergency_report",
            "disaster_type": disaster_type,
            "location": location,
            "severity": severity,
            "content": reports.get(disaster_type, "Disaster event detected"),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "automated_disaster_detection_system",
        }
    
    @staticmethod
    def generate_audio_log(disaster_type: str, location: str) -> Dict[str, Any]:
        log_contents = {
            "earthquake": f"Radio dispatch: Receiving multiple emergency calls from {location}. Structural damage reported. Medical assistance needed. Rescue teams mobilizing.",
            "flood": f"Emergency hotline recording: Citizens reporting property damage in {location}. Current water level exceeding historical average. Sandbag distribution ongoing.",
            "wildfire": f"Incident command audio: Fire spreading toward populated areas in {location}. Evacuation coordination with local authorities. Air tankers responding.",
            "hurricane": f"Weather radar confirmation: Hurricane tracking toward {location}. Storm surge predictions critical. Emergency operations center activated.",
            "landslide": f"Field report: Witnesses describing massive earth movement in {location}. Potential secondary slides. Monitoring equipment deployed.",
            "tsunami": f"Seismic monitoring: Tsunami wave confirmed approaching {location}. Propagation speed critical. Coastal facilities on high alert.",
        }
        
        return {
            "type": "audio_emergency_log",
            "disaster_type": disaster_type,
            "location": location,
            "content": log_contents.get(disaster_type, "Emergency communication"),
            "duration_seconds": np.random.randint(30, 300),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "emergency_radio_network",
            "clarity_index": np.random.uniform(0.7, 1.0),
        }
    
    @staticmethod
    def generate_image_metadata(disaster_type: str, location: str, severity: int) -> Dict[str, Any]:
        content_descriptions = {
            "earthquake": f"Satellite imagery showing structural damage patterns in {location}. Building collapse indicators. Infrastructure disruption visible.",
            "flood": f"Aerial photography of {location} showing water inundation extent. Submerged vehicles visible. Agricultural land underwater.",
            "wildfire": f"Thermal imaging of {location} showing active fire perimeter. Smoke plume visible. Surrounding vegetation risk assessment.",
            "hurricane": f"Weather satellite view of {location} with hurricane system overhead. Wind field visualization. Storm structure analysis.",
            "landslide": f"Topographic analysis of {location} showing terrain displacement. Scarp formation visible. Risk zone delineation.",
            "tsunami": f"Coastal imagery of {location} showing inundation extent. Debris field visible. Wave action captured.",
        }
        
        return {
            "type": "satellite_image",
            "disaster_type": disaster_type,
            "location": location,
            "severity": severity,
            "description": content_descriptions.get(disaster_type, "Disaster event imagery"),
            "resolution_meters": np.random.randint(1, 10),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "satellite_earth_observation",
            "analysis_confidence": np.random.uniform(0.75, 0.99),
        }
    
    @staticmethod
    def generate_multimodal_event(disaster_type: str, 
                                 location: str, 
                                 severity: int) -> Dict[str, Any]:
        return {
            "event_id": np.random.randint(10000, 99999),
            "disaster_type": disaster_type,
            "location": location,
            "severity": severity,
            "modalities": ["text", "image", "audio"],
            "text_report": DisasterDataGenerator.generate_text_report(disaster_type, location, severity),
            "image_metadata": DisasterDataGenerator.generate_image_metadata(disaster_type, location, severity),
            "audio_log": DisasterDataGenerator.generate_audio_log(disaster_type, location),
            "timestamp": datetime.utcnow().isoformat(),
            "integrated_analysis": f"Multi-modal disaster event: {disaster_type} at {location} with severity {severity}",
        }
    
    @staticmethod
    def generate_dataset(num_events: int = 20) -> List[Dict[str, Any]]:
        import random
        dataset = []
        
        for _ in range(num_events):
            disaster_type = random.choice(DisasterDataGenerator.DISASTER_TYPES)
            location = random.choice(DisasterDataGenerator.LOCATIONS)
            severity = random.choice(DisasterDataGenerator.SEVERITY_LEVELS)
            
            event = DisasterDataGenerator.generate_multimodal_event(
                disaster_type, location, severity
            )
            dataset.append(event)
        
        return dataset


class DataIngestionPipeline:
    def __init__(self, qdrant_memory, embedder, evolving_memory):
        self.qdrant_memory = qdrant_memory
        self.embedder = embedder
        self.evolving_memory = evolving_memory
    
    def ingest_multimodal_event(self, event: Dict[str, Any]) -> Dict[str, int]:
        point_ids = {}
        
        if "text_report" in event:
            text_data = event["text_report"]
            text_vector = self.embedder.embed_text(text_data.get("content", ""))
            
            metadata = {
                "modalities": ["text"],
                "disaster_type": event.get("disaster_type"),
                "severity": event.get("severity"),
                "location": event.get("location"),
                "source_type": text_data.get("source", "unknown"),
                "content_summary": text_data.get("content", "")[:200],
            }
            
            text_id = self.qdrant_memory.store_point(text_vector, metadata)
            point_ids["text"] = text_id
            
            memory_id = f"text_{text_id}"
            self.evolving_memory.store_memory(
                memory_id, text_vector, metadata
            )
        
        if "image_metadata" in event:
            image_data = event["image_metadata"]
            image_vector = np.random.randn(512).astype(np.float32)
            if "description" in image_data:
                image_vector = self.embedder.embed_text(image_data["description"])
            
            metadata = {
                "modalities": ["image"],
                "disaster_type": event.get("disaster_type"),
                "severity": event.get("severity"),
                "location": event.get("location"),
                "source_type": image_data.get("source", "unknown"),
                "content_summary": image_data.get("description", "")[:200],
                "analysis_confidence": image_data.get("analysis_confidence", 0),
            }
            
            image_id = self.qdrant_memory.store_point(image_vector, metadata)
            point_ids["image"] = image_id
            
            memory_id = f"image_{image_id}"
            self.evolving_memory.store_memory(
                memory_id, image_vector, metadata
            )
        
        if "audio_log" in event:
            audio_data = event["audio_log"]
            audio_vector = np.random.randn(512).astype(np.float32)
            if "content" in audio_data:
                audio_vector = self.embedder.embed_text(audio_data["content"])
            
            metadata = {
                "modalities": ["audio"],
                "disaster_type": event.get("disaster_type"),
                "severity": event.get("severity"),
                "location": event.get("location"),
                "source_type": audio_data.get("source", "unknown"),
                "content_summary": audio_data.get("content", "")[:200],
                "clarity_index": audio_data.get("clarity_index", 0),
            }
            
            audio_id = self.qdrant_memory.store_point(audio_vector, metadata)
            point_ids["audio"] = audio_id
            
            memory_id = f"audio_{audio_id}"
            self.evolving_memory.store_memory(
                memory_id, audio_vector, metadata
            )
        
        return point_ids
    
    def batch_ingest(self, events: List[Dict[str, Any]]) -> List[Dict[str, int]]:
        return [self.ingest_multimodal_event(event) for event in events]
