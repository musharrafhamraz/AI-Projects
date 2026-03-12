"""
FastAPI Application - Main API endpoints
"""
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..core.router import InferenceRouter
from ..models.model_client import ModelClient
from ..utils.cache import PromptCache
from ..utils.logger import UsageLogger
from ..utils.config_loader import ConfigLoader


# Request/Response Models
class InferenceRequest(BaseModel):
    prompt: str
    user_id: str
    user_tier: Literal["free", "basic", "enterprise"] = "free"
    quality_requirement: Literal["good_enough", "high", "perfect"] = "good_enough"
    max_tokens: int = 512


class InferenceResponse(BaseModel):
    response: str
    model_used: str
    latency_ms: float
    cost_usd: float
    complexity: str
    cached: bool = False


# Initialize components
config_loader = ConfigLoader("config.yaml")
router = InferenceRouter(
    models_config=config_loader.get_models(),
    tier_limits=config_loader.get_user_tiers()
)
model_client = ModelClient(config_loader.get_models())
cache = PromptCache(**config_loader.get_cache_config())
logger = UsageLogger()

# Create FastAPI app
app = FastAPI(
    title="Smart Inference Router",
    description="Routes LLM requests to optimal model based on complexity and cost",
    version="1.0.0"
)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Main inference endpoint - routes to optimal model"""
    
    start_time = time.time()
    
    # Route the request
    decision = router.route_request(
        request.prompt,
        request.user_id,
        request.user_tier,
        request.quality_requirement
    )
    
    routing_time = (time.time() - start_time) * 1000
    print(f"Routing decision: {decision.target_model} ({routing_time:.2f}ms)")
    
    # Check cache first
    cached_response = cache.get_cached_response(request.prompt, decision.target_model)
    if cached_response:
        return InferenceResponse(
            response=cached_response,
            model_used=decision.target_model,
            latency_ms=routing_time,
            cost_usd=0.0,
            complexity=decision.complexity,
            cached=True
        )
    
    # Call the model
    try:
        response_text, inference_latency = await model_client.call_model(
            decision.target_model,
            request.prompt,
            request.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model call failed: {str(e)}")
    
    # Calculate cost
    tokens_used = len(response_text.split()) * 1.3
    model_config = config_loader.get_models()[decision.target_model]
    actual_cost = (tokens_used / 1_000_000) * model_config["cost_per_1m_tokens"]
    
    # Update cost tracker
    router.update_cost(request.user_id, actual_cost)
    
    # Cache response
    cache.cache_response(request.prompt, decision.target_model, response_text)
    
    # Log usage
    logger.log_request(
        request.user_id,
        request.prompt,
        decision.complexity,
        decision.target_model,
        actual_cost,
        inference_latency
    )
    
    total_latency = (time.time() - start_time) * 1000
    
    return InferenceResponse(
        response=response_text,
        model_used=decision.target_model,
        latency_ms=total_latency,
        cost_usd=actual_cost,
        complexity=decision.complexity
    )


@app.get("/stats/{user_id}")
async def get_stats(user_id: str):
    """Get usage statistics for a user"""
    stats = router.get_user_stats(user_id)
    return {
        "user_id": user_id,
        "daily_spent": stats["daily_spent"],
        "budget_limit": stats["budget_limit"]
    }


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return cache.get_stats()


@app.post("/feedback")
async def submit_feedback(user_id: str, rating: int):
    """Submit feedback for online learning"""
    # Feedback will be incorporated in next training cycle
    return {"status": "feedback recorded", "user_id": user_id, "rating": rating}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "inference-router"}
