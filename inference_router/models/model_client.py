"""
Model Client - Handles communication with different model endpoints
"""
from typing import Optional
import httpx


class ModelClient:
    """Unified client for calling different model types"""
    
    def __init__(self, models_config: dict):
        self.models = models_config
    
    async def call_model(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int
    ) -> tuple[str, float]:
        """Call the specified model and return response + latency"""
        
        model_config = self.models[model_name]
        
        if model_config["type"] == "vllm":
            return await self._call_vllm(model_config, prompt, max_tokens)
        elif model_config["type"] == "api":
            return await self._call_api(model_name, model_config, prompt, max_tokens)
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
    
    async def _call_vllm(
        self,
        config: dict,
        prompt: str,
        max_tokens: int
    ) -> tuple[str, float]:
        """Call vLLM cluster endpoint"""
        import time
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                config["endpoint"],
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["text"]
            
        latency = (time.time() - start_time) * 1000
        return result, latency
    
    async def _call_api(
        self,
        model_name: str,
        config: dict,
        prompt: str,
        max_tokens: int
    ) -> tuple[str, float]:
        """Call external API (OpenAI, Anthropic, etc.)"""
        import time
        import os
        
        start_time = time.time()
        
        # Get API key from environment
        api_key = os.getenv(config.get("api_key_env", ""))
        
        headers = self._build_headers(config["endpoint"], api_key)
        payload = self._build_payload(config["endpoint"], prompt, max_tokens)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                config["endpoint"],
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = self._extract_response(config["endpoint"], response.json())
        
        latency = (time.time() - start_time) * 1000
        return result, latency
    
    def _build_headers(self, endpoint: str, api_key: Optional[str]) -> dict:
        """Build headers for API request"""
        if not api_key:
            return {}
        
        if "anthropic" in endpoint:
            return {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
        else:
            return {"Authorization": f"Bearer {api_key}"}
    
    def _build_payload(self, endpoint: str, prompt: str, max_tokens: int) -> dict:
        """Build payload for API request"""
        if "anthropic" in endpoint:
            return {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            return {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
    
    def _extract_response(self, endpoint: str, data: dict) -> str:
        """Extract response text from API response"""
        if "anthropic" in endpoint:
            return data["content"][0]["text"]
        else:
            return data["choices"][0]["message"]["content"]
