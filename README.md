# Smart Inference Router

A production-ready LLM traffic cop that routes requests to the optimal model based on complexity, cost, and quality requirements. Makes routing decisions in <10ms and saves 10-100x on inference costs.

## Features

- **Ultra-fast routing** (<10ms decision time)
- **Smart classification** (simple/medium/hard complexity)
- **Cost optimization** (routes to cheapest model that meets requirements)
- **Multi-tier support** (free, basic, enterprise users)
- **Budget tracking** (daily spending limits)
- **Prompt caching** (Redis-based KV cache reuse)
- **Online learning** (improves classifier from usage logs)

## Project Structure

```
inference_router/
├── api/                    # FastAPI endpoints
├── core/                   # Routing & classification logic
├── models/                 # Model client implementations
└── utils/                  # Cache, config, logging

scripts/                    # Testing & training scripts
config/                     # Configuration files
docs/                       # Documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r config/requirements.txt

# Configure models (edit config/config.yaml)
# Set your model endpoints and API keys

# Start the service
python main.py
```

Server runs on `http://localhost:8080`

## Usage

```python
import httpx

response = httpx.post("http://localhost:8080/infer", json={
    "prompt": "Write a Python function to sort a list",
    "user_id": "user123",
    "user_tier": "basic",
    "quality_requirement": "good_enough",
    "max_tokens": 512
})

result = response.json()
print(f"Model: {result['model_used']}")
print(f"Cost: ${result['cost_usd']:.6f}")
```

## Routing Examples

| Query | Complexity | User Tier | Model Selected | Cost/1M |
|-------|-----------|-----------|----------------|---------|
| "Hello" | Simple | Free | Llama-8B | $0.01 |
| "Sort a list" | Medium | Basic | Llama-8B | $0.01 |
| "Financial analysis" | Hard | Enterprise | Claude-4 | $15.00 |

## Testing

```bash
# Benchmark routing speed
python scripts/benchmark.py

# Test API endpoints
python scripts/test_api.py
```

## Training

```bash
# Retrain classifier on usage logs
python scripts/train_classifier.py
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Quick Start](docs/QUICKSTART.md) - Detailed setup guide
- [Full Documentation](docs/README.md) - Complete reference

## License

MIT
