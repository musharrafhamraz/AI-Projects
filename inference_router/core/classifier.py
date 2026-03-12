"""
Request Classifier - Determines query complexity
"""
from typing import Literal
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RequestClassifier:
    """Classifies requests into complexity levels in <5ms"""
    
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self._load_ml_classifier(model_path)
        
        # Keyword-based fallback for ultra-fast classification
        self.simple_keywords = {"hello", "hi", "thanks", "bye", "what is", "define"}
        self.hard_keywords = {
            "analyze", "compare", "research", "evaluate", "financial", 
            "comprehensive", "detailed analysis", "in-depth"
        }
    
    def _load_ml_classifier(self, model_path: str):
        """Load fine-tuned ML classifier"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded ML classifier from {model_path}")
        except Exception as e:
            print(f"Could not load ML classifier: {e}")
    
    def classify_fast(self, prompt: str) -> Literal["simple", "medium", "hard"]:
        """Ultra-fast classification using rules + optional ML"""
        
        # Try ML classifier first if available
        if self.model and self.tokenizer:
            try:
                return self._classify_with_ml(prompt)
            except Exception:
                pass  # Fall back to rules
        
        # Rule-based classification
        return self._classify_with_rules(prompt)
    
    def _classify_with_rules(self, prompt: str) -> Literal["simple", "medium", "hard"]:
        """Rule-based classification"""
        prompt_lower = prompt.lower()[:200]
        
        # Simple check
        if any(kw in prompt_lower for kw in self.simple_keywords):
            return "simple"
        
        # Hard check
        if any(kw in prompt_lower for kw in self.hard_keywords):
            return "hard"
        
        # Length-based heuristic
        if len(prompt) < 50:
            return "simple"
        elif len(prompt) > 500:
            return "hard"
        
        return "medium"
    
    def _classify_with_ml(self, prompt: str) -> Literal["simple", "medium", "hard"]:
        """ML-based classification"""
        inputs = self.tokenizer(
            prompt[:128],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        label_map = {0: "simple", 1: "medium", 2: "hard"}
        return label_map[prediction]
