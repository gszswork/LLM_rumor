"""
ToG Reasoning Module for Fact-Checking System
This module adapts the Think-on-Graph reasoning approach for fact-checking claims.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv
import os
from wikidata_client import QueryResult
from sparql_generator import SPARQLQuery

load_dotenv()

@dataclass
class FactCheckResult:
    """Represents the result of fact-checking a claim"""
    claim: str
    verdict: str  # 'SUPPORTED', 'REFUTED', 'INSUFFICIENT_EVIDENCE'
    confidence: float
    evidence: List[Dict[str, Any]]
    reasoning_chain: List[Dict[str, Any]]
    explanation: str

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain"""
    step_type: str  # 'entity_search', 'property_check', 'comparison', 'conclusion'
    description: str
    evidence: List[Dict[str, Any]]
    confidence: float
    result: Any

class ToGReasoner:
    """Adapts ToG reasoning approach for fact-checking"""
    
    def __init__(self):
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Reasoning parameters
        self.max_reasoning_depth = 3
        self.confidence_threshold = 0.7
        self.evidence_threshold = 2  # Minimum pieces of evidence needed
    
    def fact_check_claim(self, claim: str, query_results: List[QueryResult]) -> FactCheckResult:
        """Main fact-checking method using ToG-style reasoning"""
        print(f"Fact-checking claim: {claim}")
        
        # Extract knowledge triplets from query results
        knowledge_triplets = self._extract_knowledge_triplets(query_results)
        print(f"Extracted {len(knowledge_triplets)} knowledge triplets")
        
        # Perform multi-step reasoning
        reasoning_chain = self._perform_reasoning(claim, knowledge_triplets)
        
        # Make final verdict
        verdict, confidence, explanation = self._make_verdict(claim, reasoning_chain, knowledge_triplets)
        
        # Collect evidence
        evidence = self._collect_evidence(reasoning_chain, knowledge_triplets)
        
        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            reasoning_chain=[step.__dict__ for step in reasoning_chain],
            explanation=explanation
        )
    
    def _extract_knowledge_triplets(self, query_results: List[QueryResult]) -> List[Dict[str, Any]]:
        """Extract knowledge triplets from SPARQL query results"""
        triplets = []
        
        for result in query_results:
            if not result.success or not result.results:
                continue
            
            for row in result.results:
                # Convert each result row into triplets
                subject = None
                
                # Try to identify the main subject
                for entity in result.query.entities:
                    for var, value in row.items():
                        if entity.lower() in str(value).lower():
                            subject = entity
                            break
                    if subject:
                        break
                
                if not subject and result.query.entities:
                    subject = result.query.entities[0]
                
                # Create triplets for each property-value pair
                for var, value in row.items():
                    if var.endswith('Label') or var == 'item':
                        continue
                    
                    # Clean up property names
                    property_name = var.replace('Label', '').replace('_', ' ')
                    
                    triplet = {
                        'subject': subject or 'unknown',
                        'predicate': property_name,
                        'object': str(value),
                        'source': 'wikidata',
                        'confidence': result.query.confidence,
                        'query_type': result.query.query_type
                    }
                    triplets.append(triplet)
        
        return triplets
    
    def _perform_reasoning(self, claim: str, knowledge_triplets: List[Dict[str, Any]]) -> List[ReasoningStep]:
        """Perform multi-step reasoning on the knowledge graph"""
        reasoning_chain = []
        
        # Step 1: Entity verification
        entity_step = self._verify_entities(claim, knowledge_triplets)
        reasoning_chain.append(entity_step)
        
        # Step 2: Property checking
        property_step = self._check_properties(claim, knowledge_triplets)
        reasoning_chain.append(property_step)
        
        # Step 3: Temporal reasoning (if applicable)
        temporal_step = self._temporal_reasoning(claim, knowledge_triplets)
        if temporal_step:
            reasoning_chain.append(temporal_step)
        
        # Step 4: Consistency checking
        consistency_step = self._check_consistency(claim, knowledge_triplets)
        reasoning_chain.append(consistency_step)
        
        return reasoning_chain
    
    def _verify_entities(self, claim: str, knowledge_triplets: List[Dict[str, Any]]) -> ReasoningStep:
        """Verify that entities mentioned in the claim exist in the knowledge graph"""
        relevant_triplets = [t for t in knowledge_triplets if t['query_type'] in ['existence', 'property']]
        
        # Use LLM to assess entity verification
        prompt = f"""
        Analyze whether the entities mentioned in this claim are properly identified in the knowledge graph.
        
        Claim: "{claim}"
        
        Knowledge Triplets:
        {json.dumps(relevant_triplets[:10], indent=2)}
        
        Assess:
        1. Are the main entities from the claim found in the triplets?
        2. Are the entities correctly identified?
        3. What is your confidence in entity identification?
        
        Respond with JSON:
        {{
            "entities_found": true/false,
            "confidence": 0.0-1.0,
            "explanation": "brief explanation"
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_data = json.loads(response.text.strip())
            
            return ReasoningStep(
                step_type='entity_verification',
                description='Verify entities exist in knowledge graph',
                evidence=relevant_triplets,
                confidence=result_data.get('confidence', 0.5),
                result=result_data
            )
        except Exception as e:
            print(f"Error in entity verification: {e}")
            return ReasoningStep(
                step_type='entity_verification',
                description='Verify entities exist in knowledge graph',
                evidence=relevant_triplets,
                confidence=0.3,
                result={'entities_found': False, 'confidence': 0.3, 'explanation': 'Error in verification'}
            )
    
    def _check_properties(self, claim: str, knowledge_triplets: List[Dict[str, Any]]) -> ReasoningStep:
        """Check if the properties/relationships in the claim match the knowledge graph"""
        relevant_triplets = [t for t in knowledge_triplets if t['query_type'] == 'property']
        
        prompt = f"""
        Analyze whether the relationships/properties mentioned in this claim are supported by the knowledge graph.
        
        Claim: "{claim}"
        
        Knowledge Triplets:
        {json.dumps(relevant_triplets[:15], indent=2)}
        
        Assess:
        1. What relationships does the claim assert?
        2. Are these relationships supported by the triplets?
        3. Are there any contradicting relationships?
        4. What is your confidence in the property matching?
        
        Respond with JSON:
        {{
            "properties_match": true/false,
            "supporting_triplets": ["list of relevant triplet indices"],
            "contradicting_triplets": ["list of contradicting triplet indices"],
            "confidence": 0.0-1.0,
            "explanation": "detailed explanation"
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_data = json.loads(response.text.strip())
            
            return ReasoningStep(
                step_type='property_checking',
                description='Check claim properties against knowledge graph',
                evidence=relevant_triplets,
                confidence=result_data.get('confidence', 0.5),
                result=result_data
            )
        except Exception as e:
            print(f"Error in property checking: {e}")
            return ReasoningStep(
                step_type='property_checking',
                description='Check claim properties against knowledge graph',
                evidence=relevant_triplets,
                confidence=0.3,
                result={'properties_match': False, 'confidence': 0.3, 'explanation': 'Error in checking'}
            )
    
    def _temporal_reasoning(self, claim: str, knowledge_triplets: List[Dict[str, Any]]) -> Optional[ReasoningStep]:
        """Perform temporal reasoning if the claim involves dates/time"""
        # Check if claim has temporal elements
        claim_lower = claim.lower()
        temporal_keywords = ['born', 'died', 'founded', 'established', 'in', 'during', 'after', 'before', 'year']
        
        if not any(keyword in claim_lower for keyword in temporal_keywords):
            return None
        
        # Find date-related triplets
        date_triplets = []
        for t in knowledge_triplets:
            if any(date_term in t['predicate'].lower() 
                  for date_term in ['date', 'birth', 'death', 'inception', 'time']):
                date_triplets.append(t)
        
        if not date_triplets:
            return None
        
        prompt = f"""
        Perform temporal reasoning for this claim using the date information from the knowledge graph.
        
        Claim: "{claim}"
        
        Date-related Triplets:
        {json.dumps(date_triplets, indent=2)}
        
        Assess:
        1. What temporal claims are made?
        2. Do the dates in the knowledge graph support these claims?
        3. Are there any temporal inconsistencies?
        4. What is your confidence in the temporal reasoning?
        
        Respond with JSON:
        {{
            "temporal_consistency": true/false,
            "supporting_dates": ["list of supporting dates"],
            "contradicting_dates": ["list of contradicting dates"],
            "confidence": 0.0-1.0,
            "explanation": "detailed temporal analysis"
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_data = json.loads(response.text.strip())
            
            return ReasoningStep(
                step_type='temporal_reasoning',
                description='Analyze temporal consistency of claim',
                evidence=date_triplets,
                confidence=result_data.get('confidence', 0.5),
                result=result_data
            )
        except Exception as e:
            print(f"Error in temporal reasoning: {e}")
            return ReasoningStep(
                step_type='temporal_reasoning',
                description='Analyze temporal consistency of claim',
                evidence=date_triplets,
                confidence=0.3,
                result={'temporal_consistency': False, 'confidence': 0.3, 'explanation': 'Error in temporal reasoning'}
            )
    
    def _check_consistency(self, claim: str, knowledge_triplets: List[Dict[str, Any]]) -> ReasoningStep:
        """Check overall consistency between claim and knowledge graph"""
        prompt = f"""
        Perform a final consistency check between the claim and all available knowledge.
        
        Claim: "{claim}"
        
        All Knowledge Triplets:
        {json.dumps(knowledge_triplets[:20], indent=2)}
        
        Assess:
        1. Overall consistency between claim and knowledge
        2. Strength of supporting evidence
        3. Presence of contradicting evidence
        4. Quality and reliability of evidence
        
        Respond with JSON:
        {{
            "overall_consistency": true/false,
            "evidence_strength": "weak"/"moderate"/"strong",
            "major_contradictions": ["list of major contradictions"],
            "confidence": 0.0-1.0,
            "reasoning": "detailed reasoning"
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_data = json.loads(response.text.strip())
            
            return ReasoningStep(
                step_type='consistency_check',
                description='Overall consistency analysis',
                evidence=knowledge_triplets,
                confidence=result_data.get('confidence', 0.5),
                result=result_data
            )
        except Exception as e:
            print(f"Error in consistency checking: {e}")
            return ReasoningStep(
                step_type='consistency_check',
                description='Overall consistency analysis',
                evidence=knowledge_triplets,
                confidence=0.3,
                result={'overall_consistency': False, 'confidence': 0.3, 'reasoning': 'Error in consistency check'}
            )
    
    def _make_verdict(self, claim: str, reasoning_chain: List[ReasoningStep], 
                     knowledge_triplets: List[Dict[str, Any]]) -> Tuple[str, float, str]:
        """Make final verdict based on reasoning chain"""
        # Collect evidence from reasoning steps
        supporting_evidence = 0
        contradicting_evidence = 0
        total_confidence = 0
        
        explanations = []
        
        for step in reasoning_chain:
            if step.result and isinstance(step.result, dict):
                total_confidence += step.confidence
                
                if step.step_type == 'entity_verification':
                    if step.result.get('entities_found', False):
                        supporting_evidence += 1
                        explanations.append(f"Entities verified: {step.result.get('explanation', '')}")
                    else:
                        contradicting_evidence += 1
                        explanations.append(f"Entity verification failed: {step.result.get('explanation', '')}")
                
                elif step.step_type == 'property_checking':
                    if step.result.get('properties_match', False):
                        supporting_evidence += 1
                        explanations.append(f"Properties match: {step.result.get('explanation', '')}")
                    else:
                        contradicting_evidence += 1
                        explanations.append(f"Properties don't match: {step.result.get('explanation', '')}")
                
                elif step.step_type == 'temporal_reasoning':
                    if step.result.get('temporal_consistency', False):
                        supporting_evidence += 1
                        explanations.append(f"Temporal consistency: {step.result.get('explanation', '')}")
                    else:
                        contradicting_evidence += 1
                        explanations.append(f"Temporal inconsistency: {step.result.get('explanation', '')}")
                
                elif step.step_type == 'consistency_check':
                    if step.result.get('overall_consistency', False):
                        supporting_evidence += 1
                        explanations.append(f"Overall consistency: {step.result.get('reasoning', '')}")
                    else:
                        contradicting_evidence += 1
                        explanations.append(f"Overall inconsistency: {step.result.get('reasoning', '')}")
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(reasoning_chain) if reasoning_chain else 0
        
        # Make verdict
        if supporting_evidence > contradicting_evidence and avg_confidence >= self.confidence_threshold:
            verdict = 'SUPPORTED'
        elif contradicting_evidence > supporting_evidence:
            verdict = 'REFUTED'
        else:
            verdict = 'INSUFFICIENT_EVIDENCE'
        
        explanation = "; ".join(explanations[:3])  # Top 3 explanations
        
        return verdict, avg_confidence, explanation
    
    def _collect_evidence(self, reasoning_chain: List[ReasoningStep], 
                         knowledge_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect evidence from reasoning chain"""
        evidence = []
        
        for step in reasoning_chain:
            step_evidence = {
                'type': step.step_type,
                'description': step.description,
                'confidence': step.confidence,
                'triplets': step.evidence[:5],  # Top 5 most relevant triplets
                'result': step.result
            }
            evidence.append(step_evidence)
        
        return evidence

def demo_tog_reasoning():
    """Demo function to test ToG reasoning"""
    reasoner = ToGReasoner()
    
    # Mock query results for demo
    from wikidata_client import QueryResult
    from sparql_generator import SPARQLQuery
    
    # Create mock data
    mock_query = SPARQLQuery(
        query="SELECT ?person ?birthDate WHERE { ?person wdt:P569 ?birthDate }",
        entities=["Barack Obama"],
        relations=["P569"],
        query_type="property",
        confidence=0.9,
        description="Birth date query"
    )
    
    mock_results = [
        {
            'person': 'Barack Obama',
            'birthDate': '1961-08-04',
            'birthPlace': 'Hawaii'
        }
    ]
    
    mock_query_result = QueryResult(
        query=mock_query,
        success=True,
        results=mock_results,
        execution_time=1.2
    )
    
    # Test fact-checking
    claim = "Barack Obama was born in Hawaii in 1961."
    result = reasoner.fact_check_claim(claim, [mock_query_result])
    
    print(f"Claim: {claim}")
    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Explanation: {result.explanation}")
    print(f"Evidence pieces: {len(result.evidence)}")

if __name__ == "__main__":
    demo_tog_reasoning()