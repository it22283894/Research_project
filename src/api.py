"""
Step 9: FastAPI Backend for Health Risk Analysis

Endpoints:
- POST /api/analyze - Analyze ingredient/food, return risks + evidence
- POST /api/transcribe - Whisper voice-to-text  
- GET /api/predictions/{ingredient} - GNN predictions lookup
- GET /api/health - Health check
"""

import os
import sys
import tempfile
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline, RAGConfig
from dose_response import (
    IngredientQuantity as DoseIngredient,
    DoseResponseResult,
    calculate_dose_response,
    adjust_risk_predictions,
    parse_ingredient_string,
    get_rda_info,
    get_all_rda_values
)

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "pre-processed", "kg")

# Load GNN predictions
GNN_PREDICTIONS_PATH = os.path.join(MODELS_DIR, "ingredient_risk_predictions.csv")
gnn_predictions_df = None

def load_gnn_predictions():
    global gnn_predictions_df
    if os.path.exists(GNN_PREDICTIONS_PATH):
        gnn_predictions_df = pd.read_csv(GNN_PREDICTIONS_PATH)
        print(f"✅ Loaded {len(gnn_predictions_df):,} GNN predictions")
    else:
        print(f"⚠️ GNN predictions not found at {GNN_PREDICTIONS_PATH}")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Food Health Risk API",
    description="API for analyzing food ingredients and predicting health risks",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    query: str
    include_explanation: bool = True

class AnalyzeResponse(BaseModel):
    ingredient: str
    risk_predictions: Dict[str, float]
    evidence: Dict[str, List[Dict]]
    explanation: str
    suggestions: List[str]

class TranscribeResponse(BaseModel):
    text: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]

# Quantity-Aware Models (V2)
class IngredientQuantity(BaseModel):
    """Single ingredient with quantity."""
    name: str
    amount: float = 100.0
    unit: str = "g"

class AnalyzeRequestV2(BaseModel):
    """Quantity-aware analysis request."""
    ingredients: List[IngredientQuantity]
    include_explanation: bool = True

class DoseResponseInfo(BaseModel):
    """Dose-response information for an ingredient."""
    ingredient: str
    amount: float
    unit: str
    pct_rda: float
    dose_response_factor: float
    risk_level: str
    rda_value: Optional[float] = None
    rda_unit: Optional[str] = None
    is_known_nutrient: bool

class AnalyzeResponseV2(BaseModel):
    """Quantity-aware analysis response."""
    ingredients: List[str]
    risk_predictions: Dict[str, float]
    base_predictions: Dict[str, float]
    dose_response_info: List[DoseResponseInfo]
    evidence: Dict[str, List[Dict]]
    explanation: str
    suggestions: List[str]
    quantity_summary: str

# ============================================================================
# Global State
# ============================================================================

rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        config = RAGConfig(
            faiss_index_path=os.path.join(MODELS_DIR, "faiss_pubmed.index"),
            faiss_metadata_path=os.path.join(MODELS_DIR, "faiss_pubmed_metadata.json"),
        )
        rag_pipeline = RAGPipeline(config)
        
        # Load FAISS index
        if os.path.exists(config.faiss_index_path):
            rag_pipeline.load_index()
        else:
            pubmed_path = os.path.join(DATA_DIR, "pubmed_kg_triples.csv")
            if os.path.exists(pubmed_path):
                rag_pipeline.build_index(pubmed_path)
    
    return rag_pipeline

# ============================================================================
# Helper Functions
# ============================================================================

def get_gnn_predictions_single(ingredient: str) -> Dict[str, float]:
    """Get GNN predictions for a single ingredient."""
    if gnn_predictions_df is None:
        return {}
    
    ingredient = ingredient.strip()
    if not ingredient:
        return {}
    
    # Case-insensitive partial match
    mask = gnn_predictions_df['ingredient'].str.lower().str.contains(
        ingredient.lower(), na=False
    )
    matches = gnn_predictions_df[mask]
    
    if len(matches) == 0:
        return {}
    
    row = matches.iloc[0]
    predictions = {}
    
    target_diseases = ['diabetes', 'obesity', 'hypertension', 'cancer', 'cardiovascular disease']
    for disease in target_diseases:
        col_name = f"{disease}_risk"
        if col_name in row:
            predictions[disease] = float(row[col_name])
    
    return predictions


def get_gnn_predictions(query: str) -> Dict[str, float]:
    """Get GNN predictions for one or more ingredients (comma-separated).
    
    For multiple ingredients, returns the maximum risk for each disease.
    """
    if gnn_predictions_df is None:
        return {}
    
    # Split by comma and clean up
    ingredients = [ing.strip() for ing in query.split(',') if ing.strip()]
    
    if not ingredients:
        return {}
    
    # If single ingredient, use simple lookup
    if len(ingredients) == 1:
        return get_gnn_predictions_single(ingredients[0])
    
    # Multiple ingredients: aggregate predictions (take max risk per disease)
    aggregated = {}
    target_diseases = ['diabetes', 'obesity', 'hypertension', 'cancer', 'cardiovascular disease']
    
    for ingredient in ingredients:
        preds = get_gnn_predictions_single(ingredient)
        for disease in target_diseases:
            if disease in preds:
                if disease not in aggregated:
                    aggregated[disease] = preds[disease]
                else:
                    # Take the maximum risk among all ingredients
                    aggregated[disease] = max(aggregated[disease], preds[disease])
    
    return aggregated

def generate_suggestions_llm(
    ingredient: str,
    risk_predictions: Dict[str, float],
    ollama_url: str = "http://localhost:11434",
    model: str = "jsk/bio-mistral"
) -> List[str]:
    """Generate dietary suggestions using LLM based on risk predictions."""
    
    if not risk_predictions:
        return [
            "✅ This ingredient appears to have low health risks",
            "🍎 Continue maintaining a balanced diet"
        ]
    
    # Sort risks by probability descending
    sorted_risks = sorted(risk_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Build risk summary for prompt
    risk_summary = "\n".join([
        f"- {disease.title()}: {prob*100:.1f}% risk ({'HIGH' if prob > 0.6 else 'MODERATE' if prob > 0.3 else 'LOW'})"
        for disease, prob in sorted_risks
    ])
    
    prompt = f"""You are a nutrition advisor. Based on the health risk analysis for "{ingredient}", provide 4-5 specific, actionable dietary suggestions.

HEALTH RISK PREDICTIONS for {ingredient}:
{risk_summary}

INSTRUCTIONS:
1. Provide practical, specific dietary recommendations
2. Address the highest risks first
3. Include both what to avoid AND healthy alternatives
4. Each suggestion should start with an emoji
5. Keep each suggestion to one concise sentence
6. Be evidence-based and practical

Return ONLY the suggestions, one per line, no numbering."""

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.5
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            text = response.json().get('response', '')
            # Parse suggestions (one per line)
            suggestions = [line.strip() for line in text.strip().split('\n') if line.strip()]
            # Ensure each has an emoji, add one if missing
            formatted = []
            for s in suggestions[:6]:  # Max 6 suggestions
                if s and not s[0].isascii() or s.startswith(('🍎', '🥗', '🏃', '🧂', '❤️', '💡', '⚠️', '✅', '🚫', '🥦', '🐟', '🍬')):
                    formatted.append(s)
                else:
                    formatted.append(f"💡 {s}")
            return formatted if formatted else _fallback_suggestions(risk_predictions)
        else:
            print(f"LLM suggestion error: {response.status_code}")
            return _fallback_suggestions(risk_predictions)
            
    except Exception as e:
        print(f"LLM suggestion error: {e}")
        return _fallback_suggestions(risk_predictions)


def _fallback_suggestions(risk_predictions: Dict[str, float]) -> List[str]:
    """Fallback rule-based suggestions if LLM fails."""
    suggestions = []
    high_risk = [d for d, p in risk_predictions.items() if p > 0.6]
    
    if 'diabetes' in high_risk:
        suggestions.append("🍬 Limit sugar intake and choose complex carbohydrates")
    if 'obesity' in high_risk:
        suggestions.append("🏃 Increase physical activity and practice portion control")
    if 'hypertension' in high_risk:
        suggestions.append("🧂 Reduce sodium intake to less than 2,300mg/day")
    if 'cardiovascular disease' in high_risk or 'cardiovascular' in high_risk:
        suggestions.append("❤️ Choose unsaturated fats and include omega-3 rich foods")
    if 'cancer' in high_risk:
        suggestions.append("🥦 Increase antioxidant-rich vegetables in your diet")
    
    if not suggestions:
        suggestions.append("✅ This ingredient appears to have low health risks")
        suggestions.append("🍎 Continue maintaining a balanced diet")
    
    return suggestions


def generate_suggestions(risk_predictions: Dict[str, float], ingredient: str = "") -> List[str]:
    """Generate dietary suggestions - uses LLM when available, falls back to rules."""
    return generate_suggestions_llm(ingredient, risk_predictions)

def transcribe_audio_whisper(audio_bytes: bytes) -> str:
    """Transcribe audio using local Whisper model with GPU acceleration."""
    try:
        import whisper
        import torch
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        # Use 'base' model for better accuracy (vs 'tiny')
        # Models: tiny, base, small, medium, large
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        
        # Transcribe with English language hint for better accuracy
        result = model.transcribe(
            temp_path,
            language="en",
            task="transcribe",
            fp16=(device == "cuda")  # Use FP16 on GPU for speed
        )
        
        # Cleanup temp file
        os.unlink(temp_path)
        
        text = result.get("text", "").strip()
        print(f"Whisper transcription: '{text}'")
        return text
        
    except ImportError as e:
        print(f"Whisper import error: {e}")
        return ""
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_gnn_predictions()
    # Pre-load RAG pipeline
    get_rag_pipeline()

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline = get_rag_pipeline()
    
    services = {
        "faiss": pipeline.faiss_index.index is not None,
        "neo4j": pipeline.neo4j.driver is not None,
        "gnn_predictions": gnn_predictions_df is not None,
        "ollama": False
    }
    
    # Check Ollama
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        services["ollama"] = resp.status_code == 200
    except:
        pass
    
    return HealthResponse(
        status="healthy" if all(services.values()) else "degraded",
        services=services
    )

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_ingredient(request: AnalyzeRequest):
    """Analyze an ingredient or food for health risks."""
    try:
        pipeline = get_rag_pipeline()
        
        # Get GNN predictions
        gnn_preds = get_gnn_predictions(request.query)
        
        # Run RAG analysis
        result = pipeline.analyze_ingredient(request.query, gnn_predictions=gnn_preds)
        
        # Generate suggestions (LLM-based)
        suggestions = generate_suggestions(gnn_preds, ingredient=request.query)
        
        # Format evidence
        evidence = {}
        for disease, ev_list in result.get('disease_evidence', {}).items():
            evidence[disease] = [
                {
                    'pmid': e.get('pmid', ''),
                    'title': e.get('title', ''),
                    'journal': e.get('journal', ''),
                    'year': e.get('year', ''),
                    'ingredient': e.get('ingredient', ''),
                }
                for e in ev_list[:5]  # Top 5 per disease
            ]
        
        return AnalyzeResponse(
            ingredient=request.query,
            risk_predictions=gnn_preds,
            evidence=evidence,
            explanation=result.get('explanation', ''),
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/v2", response_model=AnalyzeResponseV2)
async def analyze_ingredient_v2(request: AnalyzeRequestV2):
    """Quantity-aware analysis: considers daily usage amounts for risk predictions."""
    try:
        pipeline = get_rag_pipeline()
        
        # Convert Pydantic models to dose_response dataclasses
        dose_ingredients = [
            DoseIngredient(name=ing.name, amount=ing.amount, unit=ing.unit)
            for ing in request.ingredients
        ]
        
        # Calculate dose-response for each ingredient
        dose_results: List[DoseResponseResult] = []
        for ing in dose_ingredients:
            dr = calculate_dose_response(ing)
            dose_results.append(dr)
        
        # Get base GNN predictions for all ingredients
        ingredient_names = [ing.name for ing in dose_ingredients]
        query_str = ", ".join(ingredient_names)
        base_preds = get_gnn_predictions(query_str)
        
        # Adjust predictions based on dose-response
        adjusted_preds = adjust_risk_predictions(base_preds, dose_results)
        
        # Build dose response info for response
        dose_info = []
        for dr in dose_results:
            info = dr.to_dict()
            dose_info.append(DoseResponseInfo(
                ingredient=info['ingredient'],
                amount=info['original_amount'],
                unit=info['original_unit'],
                pct_rda=info['pct_rda'],
                dose_response_factor=info['dose_response_factor'],
                risk_level=info['risk_level'],
                rda_value=info['rda_value'],
                rda_unit=info['rda_unit'],
                is_known_nutrient=info['is_known_nutrient']
            ))
        
        # Build quantity summary for explanations
        quantity_parts = []
        for dr in dose_results:
            if dr.is_known_nutrient:
                quantity_parts.append(
                    f"{dr.ingredient}: {dr.original_amount}{dr.original_unit} "
                    f"({dr.pct_rda*100:.0f}% of daily limit/target)"
                )
            else:
                quantity_parts.append(f"{dr.ingredient}: {dr.original_amount}{dr.original_unit}")
        quantity_summary = "; ".join(quantity_parts)
        
        # Run RAG analysis with quantity context
        result = pipeline.analyze_ingredient(
            query_str, 
            gnn_predictions=adjusted_preds,
            quantity_context=quantity_summary
        )
        
        # Generate suggestions with quantity awareness
        suggestions = generate_suggestions(adjusted_preds, ingredient=query_str)
        
        # Format evidence
        evidence = {}
        for disease, ev_list in result.get('disease_evidence', {}).items():
            evidence[disease] = [
                {
                    'pmid': e.get('pmid', ''),
                    'title': e.get('title', ''),
                    'journal': e.get('journal', ''),
                    'year': e.get('year', ''),
                    'ingredient': e.get('ingredient', ''),
                }
                for e in ev_list[:5]
            ]
        
        return AnalyzeResponseV2(
            ingredients=ingredient_names,
            risk_predictions=adjusted_preds,
            base_predictions=base_preds,
            dose_response_info=dose_info,
            evidence=evidence,
            explanation=result.get('explanation', ''),
            suggestions=suggestions,
            quantity_summary=quantity_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/text")
async def analyze_text_with_quantities(request: AnalyzeRequest):
    """Parse free-form text like '50g sugar, 2000mg sodium' and analyze with quantities."""
    try:
        # Parse the text input into structured ingredients
        parsed = parse_ingredient_string(request.query)
        
        if not parsed:
            raise HTTPException(status_code=400, detail="Could not parse ingredients from text")
        
        # Convert to AnalyzeRequestV2 format
        ingredients = [
            IngredientQuantity(name=p.name, amount=p.amount, unit=p.unit)
            for p in parsed
        ]
        
        # Reuse v2 endpoint logic
        v2_request = AnalyzeRequestV2(
            ingredients=ingredients,
            include_explanation=request.include_explanation
        )
        
        return await analyze_ingredient_v2(v2_request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rda-values")
async def get_rda_values():
    """Get all RDA reference values for frontend display."""
    return {
        "rda_values": get_all_rda_values()
    }

@app.get("/api/predictions/{ingredient}")
async def get_predictions(ingredient: str):
    """Get GNN predictions for an ingredient."""
    predictions = get_gnn_predictions(ingredient)
    
    if not predictions:
        raise HTTPException(status_code=404, detail=f"No predictions found for '{ingredient}'")
    
    return {
        "ingredient": ingredient,
        "predictions": predictions,
        "risk_levels": {
            disease: "high" if prob > 0.6 else "moderate" if prob > 0.3 else "low"
            for disease, prob in predictions.items()
        }
    }

@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text using Whisper."""
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # Transcribe
        text = transcribe_audio_whisper(audio_bytes)
        
        return TranscribeResponse(
            text=text,
            confidence=0.95 if text else 0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diseases")
async def get_diseases():
    """Get list of target diseases."""
    return {
        "diseases": [
            {"id": "diabetes", "name": "Diabetes", "color": "#FF6384"},
            {"id": "obesity", "name": "Obesity", "color": "#36A2EB"},
            {"id": "hypertension", "name": "Hypertension", "color": "#FFCE56"},
            {"id": "cancer", "name": "Cancer", "color": "#4BC0C0"},
            {"id": "cardiovascular disease", "name": "Cardiovascular Disease", "color": "#9966FF"},
        ]
    }

@app.get("/api/top-risk-ingredients")
async def get_top_risk_ingredients(disease: str = "diabetes", limit: int = 10):
    """Get top risk ingredients for a disease."""
    if gnn_predictions_df is None:
        raise HTTPException(status_code=503, detail="GNN predictions not loaded")
    
    col_name = f"{disease}_risk"
    if col_name not in gnn_predictions_df.columns:
        raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")
    
    top = gnn_predictions_df.nlargest(limit, col_name)[['ingredient', col_name]]
    
    return {
        "disease": disease,
        "ingredients": [
            {"name": row['ingredient'], "risk": float(row[col_name])}
            for _, row in top.iterrows()
        ]
    }

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
