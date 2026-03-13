"""
Test API endpoints
"""
import asyncio
import httpx


async def test_inference_api():
    """Test the inference router API"""
    
    base_url = "http://localhost:8080"
    
    test_cases = [
        {
            "name": "Simple greeting",
            "prompt": "Hello, how are you?",
            "user_tier": "free"
        },
        {
            "name": "Code task",
            "prompt": "Write a Python function to sort a list",
            "user_tier": "basic"
        },
        {
            "name": "Hard analysis",
            "prompt": "Analyze Q4 financial performance vs competitors",
            "user_tier": "enterprise"
        }
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}: {test['name']}")
            print(f"{'='*60}")
            
            try:
                response = await client.post(
                    f"{base_url}/infer",
                    json={
                        "prompt": test["prompt"],
                        "user_id": f"test_user_{i}",
                        "user_tier": test["user_tier"],
                        "quality_requirement": "good_enough",
                        "max_tokens": 256
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✓ Model: {result['model_used']}")
                    print(f"  Complexity: {result['complexity']}")
                    print(f"  Latency: {result['latency_ms']:.2f}ms")
                    print(f"  Cost: ${result['cost_usd']:.6f}")
                else:
                    print(f"✗ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"✗ Exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_inference_api())
