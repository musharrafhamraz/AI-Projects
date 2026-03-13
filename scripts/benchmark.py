"""
Benchmark routing speed and accuracy
"""
import time
import sys
sys.path.insert(0, '.')

from inference_router.core.router import InferenceRouter
from inference_router.utils.config_loader import ConfigLoader


def benchmark_routing_speed():
    """Measure routing decision latency"""
    
    config = ConfigLoader("config.yaml")
    router = InferenceRouter(
        models_config=config.get_models(),
        tier_limits=config.get_user_tiers()
    )
    
    test_prompts = [
        "Hello",
        "Write a Python function to sort a list",
        "Analyze comprehensive financial implications of blockchain"
    ]
    
    print("="*60)
    print("Routing Speed Benchmark")
    print("="*60)
    
    for prompt in test_prompts:
        times = []
        for _ in range(100):
            start = time.perf_counter()
            decision = router.route_request(prompt, "bench_user", "basic", "good_enough")
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"\nPrompt: {prompt[:40]}...")
        print(f"  Avg: {avg_time:.3f}ms")
        print(f"  Min: {min(times):.3f}ms, Max: {max(times):.3f}ms")
        print(f"  Model: {decision.target_model}")


if __name__ == "__main__":
    benchmark_routing_speed()
