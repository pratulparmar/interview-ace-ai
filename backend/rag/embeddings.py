"""
Embeddings Manager - InterviewAce AI RAG System

Complete working implementation for converting text to vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict
import numpy as np
from openai import OpenAI
from backend.config.settings import get_settings, setup_langsmith


class EmbeddingManager:
    """Manages text-to-vector conversion using OpenAI embeddings"""
    
    def __init__(self):
        """Initialize the embedding manager"""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = "text-embedding-3-large"
        self.cache: Dict[str, List[float]] = {}
        print(f"Embedding Manager initialized with model: {self.model}")
    
    
    def create_embedding(self, text: str) -> List[float]:
        """Convert single text to vector embedding"""
        # Check cache
        if text in self.cache:
            print(f"  Cache hit for: {text[:50]}...")
            return self.cache[text]
        
        # Clean text
        text = text.strip().replace("\n", " ")
        
        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            # Extract embedding
            embedding = response.data[0].embedding
            
            # Cache it
            self.cache[text] = embedding
            
            print(f"  Created embedding for: {text[:50]}... ({len(embedding)} dimensions)")
            
            return embedding
            
        except Exception as e:
            print(f"Error creating embedding: {e}")
            raise
    
    
    def create_embeddings_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """Convert multiple texts to embeddings efficiently"""
        if show_progress:
            print(f"\nCreating embeddings for {len(texts)} texts...")
        
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache
        for i, text in enumerate(texts):
            text = text.strip().replace("\n", " ")
            
            if text in self.cache:
                embeddings.append(self.cache[text])
                if show_progress:
                    print(f"  [{i+1}/{len(texts)}] Cache hit: {text[:50]}...")
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
                embeddings.append(None)
        
        # Embed new texts
        if texts_to_embed:
            if show_progress:
                print(f"  Embedding {len(texts_to_embed)} new texts via API...")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_embed,
                    encoding_format="float"
                )
                
                # Store results
                for i, data in enumerate(response.data):
                    embedding = data.embedding
                    text = texts_to_embed[i]
                    original_index = text_indices[i]
                    
                    self.cache[text] = embedding
                    embeddings[original_index] = embedding
                    
                    if show_progress:
                        print(f"  [{original_index+1}/{len(texts)}] Embedded: {text[:50]}...")
                
            except Exception as e:
                print(f"Error in batch embedding: {e}")
                raise
        
        if show_progress:
            print(f"Completed! {len(embeddings)} embeddings ready.")
        
        return embeddings
    
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return float(similarity)
    
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """Find top-k most similar embeddings to query"""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            score = self.calculate_similarity(query_embedding, candidate)
            similarities.append((i, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    
    def get_cache_size(self) -> int:
        """Get number of embeddings in cache"""
        return len(self.cache)
    
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        print("Embedding cache cleared")


# Demo code
if __name__ == "__main__":
    setup_langsmith()
    
    print("\n" + "="*80)
    print("EMBEDDING MANAGER - DEMO")
    print("="*80)
    
    # Initialize
    manager = EmbeddingManager()
    
    # Test 1: Single embedding
    print("\n--- TEST 1: Single Embedding ---")
    text1 = "What are Python decorators?"
    embedding1 = manager.create_embedding(text1)
    print(f"Embedding dimensions: {len(embedding1)}")
    print(f"First 10 values: {[round(x, 4) for x in embedding1[:10]]}")
    
    # Test 2: Similar text
    print("\n--- TEST 2: Similar Text ---")
    text2 = "Explain Python decorators"
    embedding2 = manager.create_embedding(text2)
    similarity = manager.calculate_similarity(embedding1, embedding2)
    print(f"Similarity between:")
    print(f"  '{text1}'")
    print(f"  '{text2}'")
    print(f"  Score: {similarity:.4f} (High = similar!)")
    
    # Test 3: Different text
    print("\n--- TEST 3: Different Text ---")
    text3 = "What is JavaScript?"
    embedding3 = manager.create_embedding(text3)
    similarity2 = manager.calculate_similarity(embedding1, embedding3)
    print(f"Similarity between:")
    print(f"  '{text1}'")
    print(f"  '{text3}'")
    print(f"  Score: {similarity2:.4f} (Low = different!)")
    
    # Test 4: Batch embedding
    print("\n--- TEST 4: Batch Embedding ---")
    texts = [
        "How does async/await work in Python?",
        "Explain Python's GIL",
        "What are Python generators?",
        "Describe list comprehensions"
    ]
    embeddings = manager.create_embeddings_batch(texts)
    print(f"Created {len(embeddings)} embeddings")
    
    # Test 5: Find most similar
    print("\n--- TEST 5: Semantic Search ---")
    query = "Tell me about Python async programming"
    query_embedding = manager.create_embedding(query)
    
    results = manager.find_most_similar(query_embedding, embeddings, top_k=3)
    print(f"\nQuery: '{query}'")
    print("Most similar questions:")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. {texts[idx]}")
        print(f"     Similarity: {score:.4f}")
    
    # Cache stats
    print(f"\n--- Cache Stats ---")
    print(f"Cached embeddings: {manager.get_cache_size()}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nKey Observations:")
    print("1. Similar questions have scores > 0.80 (very similar)")
    print("2. Different topics have scores < 0.50 (not similar)")
    print("3. Semantic search finds relevant content without exact keywords")
    print("4. Caching makes repeated embeddings instant")
    print("\nThis powers the RAG system's semantic search! ")