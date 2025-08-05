"""
Advanced Confidence Scoring System
"""
from typing import Dict, List

class AdvancedConfidenceScorer:
    """Calculate multi-factor confidence scores"""
    
    def calculate(self, answer: str, context: str, sources: List[Dict], question: str) -> Dict[str, float]:
        """Calculate comprehensive confidence score"""
        factors = {
            'content_quality': self._assess_content_quality(answer),
            'source_reliability': self._assess_source_reliability(sources),
            'answer_completeness': self._assess_completeness(answer, question),
            'factual_consistency': self._assess_consistency(answer, context),
            'domain_relevance': self._assess_domain_relevance(answer)
        }
        
        weights = {
            'content_quality': 0.2,
            'source_reliability': 0.3,
            'answer_completeness': 0.2,
            'factual_consistency': 0.2,
            'domain_relevance': 0.1
        }
        
        overall = sum(factors[f] * weights[f] for f in factors)
        
        return {
            'overall_confidence': min(overall, 1.0),
            'factors': factors
        }