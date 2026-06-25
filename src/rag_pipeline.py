"""
Step 8: RAG Pipeline for Evidence-Based Health Risk Explanations

This module implements a Retrieval-Augmented Generation (RAG) pipeline that:
1. Uses FAISS for semantic search over PubMed evidence
2. Queries Neo4j knowledge graph for ingredient-disease relationships  
3. Generates human-readable explanations using BioMistral-7B via Ollama

Requirements:
- FAISS (faiss-gpu for GPU acceleration)
- sentence-transformers for embeddings
- neo4j driver for graph queries
- Ollama with BioMistral-7B model running in Docker
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

# FAISS and embeddings
import faiss
from sentence_transformers import SentenceTransformer

# Neo4j
from neo4j import GraphDatabase

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    # Neo4j settings
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sakuni200211"
    
    # Ollama settings (Docker)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "jsk/bio-mistral"  # BioMistral-7B
    
    # Embedding model (for FAISS)
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, good for medical text
    
    # FAISS settings
    faiss_index_path: str = "models/faiss_pubmed.index"
    faiss_metadata_path: str = "models/faiss_pubmed_metadata.json"
    
    # Search settings
    top_k_faiss: int = 10
    top_k_neo4j: int = 20
    
    # Target diseases (matching Step 7)
    target_diseases: List[str] = None
    
    def __post_init__(self):
        if self.target_diseases is None:
            self.target_diseases = [
                'diabetes',
                'obesity',
                'hypertension',
                'cancer',
                'cardiovascular'
            ]


# ============================================================================
# FAISS Index Builder
# ============================================================================

class FAISSIndexBuilder:
    """Build and query FAISS vector index from PubMed evidence."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embedding_model = None
        self.index = None
        self.metadata = []
        
    def _load_embedding_model(self):
        """Load sentence transformer model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            # Use GPU if available
            if hasattr(self.embedding_model, '_target_device'):
                print(f"  Using device: {self.embedding_model._target_device}")
        return self.embedding_model
    
    def build_index(self, pubmed_csv_path: str, save: bool = True) -> None:
        """
        Build FAISS index from PubMed triples CSV.
        
        Args:
            pubmed_csv_path: Path to pubmed_kg_triples.csv
            save: Whether to save the index to disk
        """
        print(f"Building FAISS index from: {pubmed_csv_path}")
        
        # Load PubMed data
        df = pd.read_csv(pubmed_csv_path)
        print(f"  Loaded {len(df):,} PubMed records")
        
        # Create text for embedding (title + subject + object for context)
        texts = []
        self.metadata = []
        
        for _, row in df.iterrows():
            # Combine relevant fields for rich embeddings
            text = f"{row.get('title', '')} | {row.get('subject', '')} relates to {row.get('object', '')}"
            texts.append(text)
            
            # Store metadata for retrieval
            self.metadata.append({
                'subject': str(row.get('subject', '')),
                'object': str(row.get('object', '')),
                'pmid': str(row.get('pmid', '')),
                'title': str(row.get('title', '')),
                'year': str(row.get('year', '')),
                'journal': str(row.get('journal', ''))
            })
        
        # Generate embeddings
        print("  Generating embeddings...")
        model = self._load_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS index
        print("  Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use GPU index if available
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, dimension)
            print("  Using GPU-accelerated FAISS index")
        except:
            self.index = faiss.IndexFlatIP(dimension)
            print("  Using CPU FAISS index")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"  Index built with {self.index.ntotal:,} vectors")
        
        # Save index and metadata
        if save:
            self.save(self.config.faiss_index_path, self.config.faiss_metadata_path)
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # For GPU index, need to transfer to CPU first
        if hasattr(self.index, 'index'):
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
            
        faiss.write_index(cpu_index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        
        print(f"  Saved index to: {index_path}")
        print(f"  Saved metadata to: {metadata_path}")
    
    def load(self, index_path: str = None, metadata_path: str = None) -> None:
        """Load FAISS index and metadata from disk."""
        index_path = index_path or self.config.faiss_index_path
        metadata_path = metadata_path or self.config.faiss_metadata_path
        
        print(f"Loading FAISS index from: {index_path}")
        cpu_index = faiss.read_index(index_path)
        
        # Try to move to GPU
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print("  Loaded index to GPU")
        except:
            self.index = cpu_index
            print("  Loaded index to CPU")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"  Loaded {len(self.metadata):,} metadata entries")
        
        # Preload embedding model to avoid lazy loading during requests
        print("  Preloading embedding model...")
        self._load_embedding_model()
        print("  ✅ Embedding model ready")
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Search for similar PubMed evidence.
        
        Args:
            query: Search query (e.g., ingredient name or disease)
            k: Number of results to return
            
        Returns:
            List of matching evidence dictionaries with similarity scores
        """
        k = k or self.config.top_k_faiss
        
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load() first.")
        
        # Encode query
        model = self._load_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results


# ============================================================================
# Neo4j Knowledge Graph Connector
# ============================================================================

class Neo4jConnector:
    """Connect to and query the Neo4j knowledge graph."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j at {self.config.neo4j_uri}")
        except Exception as e:
            print(f"⚠️ Could not connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def get_ingredient_diseases(self, ingredient: str, limit: int = None) -> List[Dict]:
        """
        Get diseases related to an ingredient with evidence.
        
        Args:
            ingredient: Ingredient name to query
            limit: Maximum number of results
            
        Returns:
            List of disease relationships with evidence metadata
        """
        if not self.driver:
            return []
        
        limit = limit or self.config.top_k_neo4j
        
        query = """
        MATCH (i:Ingredient {name: $ingredient})-[r:RELATES_TO]->(d:Disease)
        RETURN d.name AS disease, 
               r.pmid AS pmid, 
               r.title AS title, 
               r.year AS year, 
               r.journal AS journal
        LIMIT $limit
        """
        
        results = []
        with self.driver.session() as session:
            records = session.run(query, ingredient=ingredient, limit=limit)
            for record in records:
                results.append({
                    'ingredient': ingredient,
                    'disease': record['disease'],
                    'pmid': record['pmid'],
                    'title': record['title'],
                    'year': record['year'],
                    'journal': record['journal']
                })
        
        return results
    
    def get_ingredient_diseases_fuzzy(self, ingredient: str, limit: int = None) -> List[Dict]:
        """
        Get diseases related to an ingredient using fuzzy matching.
        
        Args:
            ingredient: Ingredient name to query (partial match)
            limit: Maximum number of results
            
        Returns:
            List of disease relationships with evidence metadata
        """
        if not self.driver:
            return []
        
        limit = limit or self.config.top_k_neo4j
        
        query = """
        MATCH (i:Ingredient)-[r:RELATES_TO]->(d:Disease)
        WHERE toLower(i.name) CONTAINS toLower($ingredient)
        RETURN i.name AS ingredient,
               d.name AS disease, 
               r.pmid AS pmid, 
               r.title AS title, 
               r.year AS year, 
               r.journal AS journal
        LIMIT $limit
        """
        
        results = []
        with self.driver.session() as session:
            records = session.run(query, ingredient=ingredient, limit=limit)
            for record in records:
                results.append({
                    'ingredient': record['ingredient'],
                    'disease': record['disease'],
                    'pmid': record['pmid'],
                    'title': record['title'],
                    'year': record['year'],
                    'journal': record['journal']
                })
        
        return results
    
    def get_foods_with_ingredient(self, ingredient: str, limit: int = 10) -> List[Dict]:
        """
        Get foods containing a specific ingredient.
        
        Args:
            ingredient: Ingredient name to query
            limit: Maximum number of results
            
        Returns:
            List of foods with the ingredient
        """
        if not self.driver:
            return []
        
        query = """
        MATCH (f:Food)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
        WHERE toLower(i.name) CONTAINS toLower($ingredient)
        RETURN f.name AS food_name, f.id AS food_id, i.name AS ingredient
        LIMIT $limit
        """
        
        results = []
        with self.driver.session() as session:
            records = session.run(query, ingredient=ingredient, limit=limit)
            for record in records:
                results.append({
                    'food_name': record['food_name'],
                    'food_id': record['food_id'],
                    'ingredient': record['ingredient']
                })
        
        return results
    
    def get_disease_statistics(self) -> Dict[str, int]:
        """Get count of ingredients linked to each disease category."""
        if not self.driver:
            return {}
        
        query = """
        MATCH (i:Ingredient)-[r:RELATES_TO]->(d:Disease)
        RETURN d.name AS disease, count(DISTINCT i) AS ingredient_count
        ORDER BY ingredient_count DESC
        LIMIT 50
        """
        
        results = {}
        with self.driver.session() as session:
            records = session.run(query)
            for record in records:
                results[record['disease']] = record['ingredient_count']
        
        return results


# ============================================================================
# Ollama LLM Client (BioMistral-7B)
# ============================================================================

class OllamaClient:
    """Client for BioMistral-7B via Ollama Docker."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.base_url = self.config.ollama_url
        self.model = self.config.ollama_model
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                if self.model in model_names or any(self.model in n for n in model_names):
                    print(f"✅ Ollama connected, model '{self.model}' available")
                else:
                    print(f"⚠️ Model '{self.model}' not found. Available: {model_names}")
                    print(f"   Run: docker exec -it ollama ollama pull {self.model}")
            else:
                print(f"⚠️ Ollama API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Cannot connect to Ollama at {self.base_url}")
            print("   Ensure Docker is running with Ollama container")
        except Exception as e:
            print(f"⚠️ Ollama connection error: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate text using BioMistral-7B.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "num_gpu": 99  # Use all GPU layers
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Ollama error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            print(f"Ollama generation error: {e}")
            return ""


# ============================================================================
# Evidence Aggregator
# ============================================================================

class EvidenceAggregator:
    """Aggregate and organize evidence from multiple sources."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
    
    def aggregate(
        self, 
        faiss_results: List[Dict], 
        neo4j_results: List[Dict],
        target_diseases: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Aggregate evidence from FAISS and Neo4j, organized by disease.
        
        Args:
            faiss_results: Results from FAISS semantic search
            neo4j_results: Results from Neo4j graph query
            target_diseases: List of target diseases to filter
            
        Returns:
            Dictionary mapping disease categories to evidence list
        """
        target_diseases = target_diseases or self.config.target_diseases
        
        # Initialize disease buckets
        disease_evidence = defaultdict(list)
        
        # Process FAISS results
        for result in faiss_results:
            disease_text = result.get('object', '').lower()
            for target in target_diseases:
                if target in disease_text:
                    evidence = {
                        'source': 'faiss',
                        'ingredient': result.get('subject', ''),
                        'disease': result.get('object', ''),
                        'pmid': result.get('pmid', ''),
                        'title': result.get('title', ''),
                        'year': result.get('year', ''),
                        'journal': result.get('journal', ''),
                        'score': result.get('similarity_score', 0)
                    }
                    disease_evidence[target].append(evidence)
        
        # Process Neo4j results
        for result in neo4j_results:
            disease_text = result.get('disease', '').lower()
            for target in target_diseases:
                if target in disease_text:
                    evidence = {
                        'source': 'neo4j',
                        'ingredient': result.get('ingredient', ''),
                        'disease': result.get('disease', ''),
                        'pmid': result.get('pmid', ''),
                        'title': result.get('title', ''),
                        'year': result.get('year', ''),
                        'journal': result.get('journal', ''),
                        'score': 1.0  # Graph edges are direct matches
                    }
                    disease_evidence[target].append(evidence)
        
        # Deduplicate by PMID
        for disease in disease_evidence:
            seen_pmids = set()
            unique_evidence = []
            for ev in disease_evidence[disease]:
                pmid = ev.get('pmid', '')
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    unique_evidence.append(ev)
            disease_evidence[disease] = unique_evidence
        
        return dict(disease_evidence)
    
    def get_evidence_summary(self, disease_evidence: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate summary statistics for aggregated evidence."""
        summary = {}
        for disease, evidence_list in disease_evidence.items():
            summary[disease] = {
                'total_evidence': len(evidence_list),
                'unique_ingredients': len(set(e['ingredient'] for e in evidence_list)),
                'year_range': self._get_year_range(evidence_list),
                'top_journals': self._get_top_journals(evidence_list, n=3)
            }
        return summary
    
    def _get_year_range(self, evidence_list: List[Dict]) -> Tuple[str, str]:
        """Get min and max years from evidence."""
        years = [e.get('year', '') for e in evidence_list if e.get('year', '').isdigit()]
        if years:
            return (min(years), max(years))
        return ('', '')
    
    def _get_top_journals(self, evidence_list: List[Dict], n: int = 3) -> List[str]:
        """Get most common journals."""
        journals = [e.get('journal', '') for e in evidence_list if e.get('journal', '')]
        from collections import Counter
        return [j for j, _ in Counter(journals).most_common(n)]


# ============================================================================
# Explanation Generator (Using BioMistral-7B)
# ============================================================================

class ExplanationGenerator:
    """Generate human-readable explanations using BioMistral-7B."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.llm = OllamaClient(config)
    
    def generate_explanation(
        self,
        ingredient: str,
        disease_evidence: Dict[str, List[Dict]],
        gnn_predictions: Dict[str, float] = None,
        quantity_context: str = None
    ) -> str:
        """
        Generate a comprehensive explanation for ingredient health risks.
        
        Args:
            ingredient: The ingredient being analyzed
            disease_evidence: Evidence organized by disease category
            gnn_predictions: Optional GNN model predictions {disease: probability}
            
        Returns:
            Human-readable explanation with citations
        """
        # Build context from evidence
        context = self._build_context(ingredient, disease_evidence, gnn_predictions)
        
        # Create prompt for BioMistral
        prompt = self._create_prompt(ingredient, context, gnn_predictions, quantity_context)
        
        # Generate explanation
        explanation = self.llm.generate(prompt, max_tokens=1024, temperature=0.3)
        
        # Add citations
        if explanation:
            explanation = self._add_citations(explanation, disease_evidence)
        else:
            # Fallback to template-based if LLM fails
            explanation = self._template_explanation(ingredient, disease_evidence, gnn_predictions)
        
        return explanation
    
    def _build_context(
        self,
        ingredient: str,
        disease_evidence: Dict[str, List[Dict]],
        gnn_predictions: Dict[str, float] = None
    ) -> str:
        """Build context string from evidence for the LLM."""
        context_parts = []
        
        for disease, evidence_list in disease_evidence.items():
            if not evidence_list:
                continue
            
            context_parts.append(f"\n{disease.upper()} Evidence:")
            
            # Add GNN prediction if available
            if gnn_predictions and disease in gnn_predictions:
                prob = gnn_predictions[disease]
                risk_level = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.4 else "LOW"
                context_parts.append(f"- GNN Risk Prediction: {prob:.1%} ({risk_level})")
            
            # Add top evidence
            for ev in evidence_list[:5]:
                title = ev.get('title', 'No title')[:100]
                journal = ev.get('journal', 'Unknown journal')
                year = ev.get('year', '')
                context_parts.append(f"- {title}... ({journal}, {year})")
        
        return "\n".join(context_parts)
    
    def _create_prompt(
        self,
        ingredient: str,
        context: str,
        gnn_predictions: Dict[str, float] = None,
        quantity_context: str = None
    ) -> str:
        """Create the prompt for BioMistral."""
        quantity_section = ""
        if quantity_context:
            quantity_section = f"""

QUANTITY CONTEXT:
User is consuming: {quantity_context}
Consider these quantities when assessing health risks - higher amounts relative to daily limits increase risk.
"""
        
        prompt = f"""You are a medical AI assistant analyzing the health impacts of food ingredients.
Based on the following scientific evidence, provide a clear, accurate explanation of the health risks 
associated with "{ingredient}" consumption.
{quantity_section}
SCIENTIFIC EVIDENCE:
{context}

INSTRUCTIONS:
1. Summarize the key health associations found in the evidence
2. Explain the biological mechanisms where possible
3. Provide practical dietary recommendations
4. Use accessible language suitable for general audiences
5. Be balanced - note both risks and any potential benefits if mentioned
6. Keep the response concise (2-3 paragraphs)
7. Write in PLAIN TEXT only - do NOT use markdown formatting (no #, *, **, etc.)
8. Use simple paragraph structure without headers or bullet points
{"9. IMPORTANT: Reference the quantity consumed and its % of daily recommended limit in your analysis" if quantity_context else ""}

PLAIN TEXT HEALTH RISK ANALYSIS FOR: {ingredient}
"""
        return prompt
    
    def _add_citations(
        self,
        explanation: str,
        disease_evidence: Dict[str, List[Dict]]
    ) -> str:
        """Add citation section to the explanation."""
        citations = []
        pmids_added = set()
        
        for disease, evidence_list in disease_evidence.items():
            for ev in evidence_list[:3]:  # Top 3 per disease
                pmid = ev.get('pmid', '')
                if pmid and pmid not in pmids_added:
                    pmids_added.add(pmid)
                    title = ev.get('title', 'No title')[:80]
                    journal = ev.get('journal', 'Unknown')
                    year = ev.get('year', '')
                    citations.append(f"[PMID:{pmid}] {title}... ({journal}, {year})")
        
        if citations:
            explanation += "\n\nReferences:\n" + "\n".join(f"• {c}" for c in citations[:10])
        
        return explanation
    
    def _template_explanation(
        self,
        ingredient: str,
        disease_evidence: Dict[str, List[Dict]],
        gnn_predictions: Dict[str, float] = None
    ) -> str:
        """Fallback template-based explanation if LLM unavailable."""
        parts = [f"Health Risk Analysis: {ingredient}\n"]
        
        # Sort diseases by risk level (high first)
        sorted_diseases = []
        for disease, evidence_list in disease_evidence.items():
            if evidence_list:
                risk_val = gnn_predictions.get(disease, 0) if gnn_predictions else 0
                sorted_diseases.append((disease, evidence_list, risk_val))
        sorted_diseases.sort(key=lambda x: x[2], reverse=True)
        
        for disease, evidence_list, _ in sorted_diseases:
            n_studies = len(evidence_list)
            
            # Risk level from GNN
            risk_text = ""
            if gnn_predictions and disease in gnn_predictions:
                prob = gnn_predictions[disease]
                risk_level = "HIGH" if prob > 0.6 else "MODERATE" if prob > 0.3 else "LOW"
                risk_text = f" - {prob:.0%} risk ({risk_level})"
            
            parts.append(f"{disease.title()}{risk_text}")
            parts.append(f"Found {n_studies} scientific studies linking {ingredient} to {disease}.")
            
            # Sample citations
            for ev in evidence_list[:2]:
                parts.append(f"  • {ev.get('title', 'Study')[:80]}... (PMID:{ev.get('pmid', 'N/A')})")
            
            parts.append("")
        
        return "\n".join(parts)


# ============================================================================
# Main RAG Pipeline
# ============================================================================

class RAGPipeline:
    """
    Main RAG Pipeline for evidence-based health risk explanations.
    
    Combines FAISS semantic search, Neo4j graph queries, and BioMistral-7B
    to generate comprehensive explanations for food ingredient health risks.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        print("Initializing RAG Pipeline...")
        self.faiss_index = FAISSIndexBuilder(self.config)
        self.neo4j = Neo4jConnector(self.config)
        self.aggregator = EvidenceAggregator(self.config)
        self.explainer = ExplanationGenerator(self.config)
        print("RAG Pipeline initialized.\n")
    
    def build_index(self, pubmed_csv_path: str) -> None:
        """Build FAISS index from PubMed data."""
        self.faiss_index.build_index(pubmed_csv_path)
    
    def load_index(self) -> None:
        """Load existing FAISS index from disk."""
        self.faiss_index.load()
    
    def analyze_ingredient(
        self,
        ingredient: str,
        gnn_predictions: Dict[str, float] = None,
        quantity_context: str = None
    ) -> Dict[str, Any]:
        """
        Analyze an ingredient and generate explanation.
        
        Args:
            ingredient: Ingredient to analyze
            gnn_predictions: Optional GNN model predictions
            quantity_context: Optional quantity info (e.g., "sugar: 50g (100% of daily limit)")
            
        Returns:
            Dictionary with evidence, summary, and explanation
        """
        print(f"Analyzing ingredient: {ingredient}")
        
        # 1. FAISS semantic search
        print("  Searching FAISS index...")
        faiss_results = self.faiss_index.search(ingredient)
        print(f"    Found {len(faiss_results)} similar evidence")
        
        # 2. Neo4j graph query
        print("  Querying Neo4j knowledge graph...")
        neo4j_results = self.neo4j.get_ingredient_diseases_fuzzy(ingredient)
        print(f"    Found {len(neo4j_results)} graph relationships")
        
        # 3. Aggregate evidence
        print("  Aggregating evidence...")
        disease_evidence = self.aggregator.aggregate(faiss_results, neo4j_results)
        evidence_summary = self.aggregator.get_evidence_summary(disease_evidence)
        
        # 4. Generate explanation (with quantity context if provided)
        print("  Generating explanation with BioMistral-7B...")
        explanation = self.explainer.generate_explanation(
            ingredient, disease_evidence, gnn_predictions, quantity_context
        )
        
        return {
            'ingredient': ingredient,
            'disease_evidence': disease_evidence,
            'evidence_summary': evidence_summary,
            'gnn_predictions': gnn_predictions,
            'explanation': explanation
        }
    
    def analyze_food(self, food_name: str) -> List[Dict[str, Any]]:
        """
        Analyze all ingredients in a food product.
        
        Args:
            food_name: Food product name to analyze
            
        Returns:
            List of ingredient analyses
        """
        # Get ingredients for the food
        foods = self.neo4j.get_foods_with_ingredient(food_name, limit=20)
        
        if not foods:
            print(f"No foods found matching: {food_name}")
            return []
        
        # Get unique ingredients
        ingredients = list(set(f['ingredient'] for f in foods))
        print(f"Found {len(ingredients)} ingredients in {food_name}")
        
        # Analyze each ingredient
        analyses = []
        for ing in ingredients[:5]:  # Limit to top 5
            analysis = self.analyze_ingredient(ing)
            analyses.append(analysis)
        
        return analyses
    
    def close(self):
        """Clean up resources."""
        self.neo4j.close()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_pipeline(
    neo4j_uri: str = "neo4j://127.0.0.1:7687",
    neo4j_password: str = "password",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "biomistral"
) -> RAGPipeline:
    """Create a configured RAG pipeline."""
    config = RAGConfig(
        neo4j_uri=neo4j_uri,
        neo4j_password=neo4j_password,
        ollama_url=ollama_url,
        ollama_model=ollama_model
    )
    return RAGPipeline(config)


def quick_analyze(ingredient: str, pipeline: RAGPipeline = None) -> str:
    """Quick analysis of an ingredient, returns just the explanation."""
    if pipeline is None:
        pipeline = create_pipeline()
        pipeline.load_index()
    
    result = pipeline.analyze_ingredient(ingredient)
    return result['explanation']


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline for Food Health Analysis")
    parser.add_argument("--ingredient", "-i", type=str, help="Ingredient to analyze")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index")
    parser.add_argument("--pubmed-csv", type=str, default="data/pre-processed/kg/pubmed_kg_triples.csv",
                        help="Path to PubMed CSV for index building")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_pipeline()
    
    if args.build_index:
        # Build index
        pipeline.build_index(args.pubmed_csv)
        print("\n✅ FAISS index built successfully!")
    
    elif args.ingredient:
        # Load index and analyze
        try:
            pipeline.load_index()
        except FileNotFoundError:
            print("Index not found. Building...")
            pipeline.build_index(args.pubmed_csv)
        
        result = pipeline.analyze_ingredient(args.ingredient)
        print("\n" + "="*60)
        print(result['explanation'])
        print("="*60)
    
    else:
        # Interactive mode
        print("RAG Pipeline Ready!")
        print("Building index from PubMed data...")
        pipeline.build_index(args.pubmed_csv)
        
        while True:
            ingredient = input("\nEnter ingredient (or 'quit'): ").strip()
            if ingredient.lower() in ['quit', 'exit', 'q']:
                break
            
            result = pipeline.analyze_ingredient(ingredient)
            print("\n" + "="*60)
            print(result['explanation'])
            print("="*60)
    
    pipeline.close()
