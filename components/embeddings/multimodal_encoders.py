import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod


class MultimodalEmbeddingGenerator(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_image(self, image_path: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_audio(self, audio_path: str) -> np.ndarray:
        pass


class UniversalSentenceEncoder(MultimodalEmbeddingGenerator):
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        try:
            import tensorflow_hub as hub
            self.text_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        except:
            self.text_model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        if self.text_model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        result = self.text_model([text])
        return np.array(result[0], dtype=np.float32)


class CLIPEmbeddingGenerator(MultimodalEmbeddingGenerator):
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        try:
            from PIL import Image
            import clip
            import torch
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            self.torch = torch
            self.Image = Image
        except:
            self.clip_model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        if self.clip_model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        tokens = self.clip_model.tokenize(text)
        with self.torch.no_grad():
            embedding = self.clip_model.encode_text(tokens)
        return embedding.numpy().astype(np.float32)[0]
    
    def embed_image(self, image_path: str) -> np.ndarray:
        if self.clip_model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        try:
            image = self.Image.open(image_path)
            preprocessed = self.clip_preprocess(image).unsqueeze(0)
            with self.torch.no_grad():
                embedding = self.clip_model.encode_image(preprocessed)
            return embedding.numpy().astype(np.float32)[0]
        except:
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def embed_audio(self, audio_path: str) -> np.ndarray:
        return np.random.randn(self.embedding_dim).astype(np.float32)


class AudioEmbeddingGenerator:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        try:
            import librosa
            import soundfile as sf
            self.librosa = librosa
            self.sf = sf
        except:
            self.librosa = None
    
    def embed_audio(self, audio_path: str) -> np.ndarray:
        if self.librosa is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        try:
            y, sr = self.librosa.load(audio_path, sr=22050)
            mfcc = self.librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.embedding_dim)
            embedding = np.mean(mfcc, axis=1)
            return (embedding / (np.linalg.norm(embedding) + 1e-8)).astype(np.float32)
        except:
            return np.random.randn(self.embedding_dim).astype(np.float32)


class HybridMultimodalEmbedder:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.clip_gen = CLIPEmbeddingGenerator(embedding_dim)
        self.audio_gen = AudioEmbeddingGenerator(embedding_dim)
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.clip_gen.embed_text(text)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        return self.clip_gen.embed_image(image_path)
    
    def embed_audio(self, audio_path: str) -> np.ndarray:
        return self.audio_gen.embed_audio(audio_path)
    
    def embed_multimodal(self, text: Optional[str] = None, 
                        image_path: Optional[str] = None,
                        audio_path: Optional[str] = None) -> np.ndarray:
        embeddings = []
        weights = []
        
        if text:
            embeddings.append(self.embed_text(text))
            weights.append(0.5)
        
        if image_path:
            embeddings.append(self.embed_image(image_path))
            weights.append(0.3)
        
        if audio_path:
            embeddings.append(self.embed_audio(audio_path))
            weights.append(0.2)
        
        if not embeddings:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        weights = np.array(weights) / sum(weights)
        combined = np.zeros(self.embedding_dim, dtype=np.float32)
        for emb, w in zip(embeddings, weights):
            combined += emb * w
        
        norm = np.linalg.norm(combined)
        return (combined / (norm + 1e-8)).astype(np.float32)
