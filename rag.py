import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, 
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 gen_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """Initialize models for retrieval and generation."""
        # 1. Setup embedding model
        self.embed_model = SentenceTransformer(embed_model_name)
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        
        # State: FAISS Index (re-initialized per session/query batch)
        self.index = None
        self.chunks = []
        self.chunk_sources = []
        
        # 2. Setup local Generator LM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(
            gen_model_name,
            torch_dtype=torch.float32,
            device_map=self.device
        )
        
    def chunk_text(self, text: str, source_url: str, chunk_size: int = 200, overlap: int = 40) -> Tuple[List[str], List[str]]:
        """Split text into overlapping word chunks."""
        words = text.split()
        chunks = []
        sources = []
        for i in range(0, len(words), max(1, chunk_size - overlap)):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                sources.append(source_url)
        return chunks, sources

    def build_index(self, documents: List[Dict[str, str]]):
        """Chunk documents and create a local FAISS index."""
        self.chunks = []
        self.chunk_sources = []
        
        for doc in documents:
            content = doc.get("content", "")
            url = doc.get("url", "")
            if content.strip():
                c, s = self.chunk_text(content, url)
                self.chunks.extend(c)
                self.chunk_sources.extend(s)
                
        if not self.chunks:
            return
            
        embeddings = self.embed_model.encode(self.chunks, convert_to_numpy=True)
        # Using inner product requires normalized vectors for cosine similarity equivalent
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks for a given query."""
        if not self.index or self.index.ntotal == 0:
            return []
            
        query_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "text": self.chunks[idx],
                    "source": self.chunk_sources[idx],
                    "score": float(distances[0][i])
                })
        return results
        
    def generate_answer(self, query: str, context: str) -> str:
        """Use local LLM to generate an answer given context."""
        prompt = f"""Use the following context to answer the user's question. If the context does not contain the answer, say "I don't have enough information to answer that based on the provided sources."

Context:
{context}

Question: {query}
Answer:"""

        # Prepare chat completion if pipeline supports it, but simple text generation works too
        # Many Instruct models work better with chat templates:
        messages = [
            {"role": "system", "content": "You are a helpful assistant answering questions strictly based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract just the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return answer.strip()
