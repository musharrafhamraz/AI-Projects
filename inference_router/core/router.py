"""
Core Routing Logic - Decides which model to use
"""
from typing import Literal
from dataclasses import dataclass
from collections import defaultdict

from .classifier import RequestClassifier


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    complexity: Literal["simple", "medium", "hard"]
    target_model: str
    endpoint_type: Literal["vllm", "api"]
    estimated_cost: float


class InferenceRouter:
    """Routes requests to optimal model based on complexity, cost, and quality"""
    
    def __init__(self, models_config: dict, tier_limits: dict):
        self.classifier = RequestClassifier()
        self.models = models_config
        self.daily_limits = tier_limits
        self.cost_tracker = defaultdict(float)
    
    def route_request(
        self,
        prompt: str,
        user_id: str,
        user_tier: str,
        quality_requirement: str
    ) -> RoutingDecision:
        """Main routing logic - decides which model to use in <10ms"""
        
        # Step 1: Fast classification
        complexity = self.classifier.classify_fast(prompt)
        
        # Step 2: Check budget
        daily_spent = self.cost_tracker[user_id]
        budget_remaining = self.daily_limits[user_tier] - daily_spent
        
        # Step 3: Apply routing rules
        target_model = self._apply_routing_rules(
            complexity, user_tier, quality_requirement, budget_remaining
        )
        
        model_config = self.models[target_model]
        
        return RoutingDecision(
            complexity=complexity,
            target_model=target_model,
            endpoint_type=model_config["type"],
            estimated_cost=model_config["cost_per_1m_tokens"] / 1_000_000 * 500
        )
    
    def _apply_routing_rules(
        self,
        complexity: str,
        user_tier: str,
        quality_requirement: str,
        budget_remaining: float
    ) -> str:
        """Core routing rules - customize for your needs"""
        
        # Rule 1: Budget exhausted → cheapest model
        if budget_remaining < 0.01:
            return "qwen-2.5-7b"
        
        # Rule 2: Simple queries → fine-tuned small model
        if complexity == "simple":
            return "llama-8b-finetune"
        
        # Rule 3: Hard queries OR perfect quality → frontier model
        if complexity == "hard" or quality_requirement == "perfect":
            if user_tier == "enterprise":
                return "claude-4"
            elif user_tier == "basic":
                return "gpt-4"
            else:
                return "llama-70b-finetune"
        
        # Rule 4: Medium complexity with high quality
        if complexity == "medium" and quality_requirement == "high":
            if user_tier in ["basic", "enterprise"]:
                return "llama-70b-finetune"
            else:
                return "llama-8b-finetune"
        
        # Default: balanced model
        return "llama-8b-finetune"
    
    def update_cost(self, user_id: str, cost: float):
        """Update user's daily cost"""
        self.cost_tracker[user_id] += cost
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get user's usage statistics"""
        return {
            "daily_spent": self.cost_tracker[user_id],
            "budget_limit": self.daily_limits.get("free", 1.0)
        }
