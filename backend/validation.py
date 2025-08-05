"""
Response Validation Module
Validates and improves LLM responses
"""
import re
from typing import Dict, Any, List

class ResponseValidator:
    """Validate and improve response quality"""
    
    def __init__(self):
        self.insurance_terms = [
            'policy', 'premium', 'coverage', 'claim', 'benefit', 'deductible',
            'exclusion', 'waiting period', 'sum assured', 'co-payment'
        ]
    
    def validate(self, response: str, context: str, question: str) -> str:
        """Validate and improve response"""
        # Basic validation
        if not response or len(response.strip()) < 10:
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        # Check for insurance relevance
        if not self._is_insurance_relevant(response):
            return response + "\n\nNote: This response may not be directly related to insurance policy terms."
        
        # Format improvements
        formatted_response = self._format_response(response)
        
        return formatted_response
    
    def _is_insurance_relevant(self, response: str) -> bool:
        """Check if response is insurance-relevant"""
        response_lower = response.lower()
        return any(term in response_lower for term in self.insurance_terms)
    
    def _format_response(self, response: str) -> str:
        """Apply formatting improvements"""
        # Add bullet points for lists
        if '\n-' in response or '\n•' in response:
            return response
        
        # Improve readability
        sentences = response.split('. ')
        if len(sentences) > 3:
            # Group related sentences
            formatted = []
            for i, sentence in enumerate(sentences):
                if i > 0 and i % 2 == 0:
                    formatted.append('\n\n')
                formatted.append(sentence.strip())
                if i < len(sentences) - 1:
                    formatted.append('. ')
            return ''.join(formatted)
        
        return response
    
    def check_confidence_factors(self, response: str, sources: List[Dict]) -> Dict[str, float]:
        """Calculate confidence factors"""
        factors = {
            'response_length': min(len(response) / 100, 1.0),
            'source_quality': len(sources) / 5.0 if sources else 0.0,
            'insurance_relevance': 1.0 if self._is_insurance_relevant(response) else 0.5,
            'specificity': self._calculate_specificity(response)
        }
        return factors
    
    def _calculate_specificity(self, response: str) -> float:
        """Calculate how specific the response is"""
        specific_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Dollar amounts
            r'\d+ days?',  # Time periods
            r'\d+ months?',  # Time periods
            r'\d+ years?',  # Time periods
        ]
        
        specificity_score = 0
        for pattern in specific_patterns:
            if re.search(pattern, response):
                specificity_score += 0.2
        
        return min(specificity_score, 1.0)