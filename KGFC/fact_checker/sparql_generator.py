"""
SPARQL Query Generator for Fact-Checking System
This module generates SPARQL queries for WikiData based on claims and extracted entities.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv
import os
from entity_extractor import Entity

load_dotenv()

@dataclass
class SPARQLQuery:
    """Represents a generated SPARQL query with metadata"""
    query: str
    entities: List[str]
    relations: List[str]
    query_type: str  # 'existence', 'property', 'count', 'comparison'
    confidence: float
    description: str

class SPARQLGenerator:
    """Generates SPARQL queries for WikiData based on claims and entities"""
    
    def __init__(self):
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Common WikiData prefixes
        self.prefixes = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://www.wikidata.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
"""
    
    def generate_sparql_query(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate SPARQL queries based on claim and entities"""
        entity_texts = [entity.text for entity in entities]
        
        prompt = f"""
        You are an expert in generating SPARQL queries for WikiData to fact-check claims.
        
        Given a claim and extracted entities, generate 1-3 SPARQL queries that would help verify or refute the claim.
        The queries should retrieve relevant information from WikiData that can be used to check the claim's accuracy.
        
        Claim: "{claim}"
        Extracted Entities: {entity_texts}
        
        Guidelines:
        1. Use standard WikiData prefixes (wd:, wdt:, rdfs:, etc.)
        2. Focus on retrieving factual information that directly relates to the claim
        3. Consider different types of queries: existence checks, property queries, count queries, date ranges
        4. Make queries specific enough to be useful but general enough to return results
        5. Include LIMIT clauses to prevent overly large result sets
        
        Return your response as a JSON array where each query object has:
        - "query": the complete SPARQL query string
        - "entities": list of entity names referenced in the query
        - "relations": list of properties/relations used (e.g., "P31", "P569")  
        - "query_type": one of "existence", "property", "count", "comparison"
        - "confidence": float 0-1 indicating confidence in query relevance
        - "description": brief description of what the query checks
        
        JSON:
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                queries_data = json.loads(json_match.group())
                sparql_queries = []
                
                for query_data in queries_data:
                    # Add prefixes to query if not present
                    query_text = query_data.get('query', '')
                    if not query_text.startswith('PREFIX'):
                        query_text = self.prefixes + "\n" + query_text
                    
                    sparql_query = SPARQLQuery(
                        query=query_text,
                        entities=query_data.get('entities', []),
                        relations=query_data.get('relations', []),
                        query_type=query_data.get('query_type', 'property'),
                        confidence=query_data.get('confidence', 0.7),
                        description=query_data.get('description', '')
                    )
                    sparql_queries.append(sparql_query)
                
                return sparql_queries
                
        except Exception as e:
            print(f"Error generating SPARQL queries: {e}")
        
        # Fallback to template-based generation
        return self.generate_template_queries(claim, entities)
    
    def generate_template_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate queries using predefined templates as fallback"""
        queries = []
        
        for entity in entities:
            if entity.label in ['PERSON', 'ORG', 'GPE']:
                # Basic entity information query
                query = f"""
{self.prefixes}

SELECT ?item ?itemLabel ?property ?value ?valueLabel WHERE {{
  ?item rdfs:label "{entity.text}"@en .
  ?item ?property ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT 50
"""
                sparql_query = SPARQLQuery(
                    query=query,
                    entities=[entity.text],
                    relations=['rdfs:label'],
                    query_type='property',
                    confidence=0.6,
                    description=f"Basic information about {entity.text}"
                )
                queries.append(sparql_query)
        
        return queries
    
    def refine_query_for_claim(self, claim: str, base_query: SPARQLQuery) -> SPARQLQuery:
        """Refine a base query to be more specific to the claim"""
        prompt = f"""
        Refine the following SPARQL query to be more specific for fact-checking this claim:
        
        Claim: "{claim}"
        Current Query: {base_query.query}
        
        Make the query more targeted to retrieve information that would specifically help verify or refute the claim.
        Focus on the key relationships and properties mentioned in the claim.
        
        Return only the refined SPARQL query:
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            refined_query = response.text.strip()
            
            # Clean up the response to extract just the query
            if 'SELECT' in refined_query:
                start_idx = refined_query.find('PREFIX')
                if start_idx == -1:
                    start_idx = refined_query.find('SELECT')
                refined_query = refined_query[start_idx:]
            
            return SPARQLQuery(
                query=refined_query,
                entities=base_query.entities,
                relations=base_query.relations,
                query_type=base_query.query_type,
                confidence=min(base_query.confidence + 0.1, 1.0),
                description=f"Refined: {base_query.description}"
            )
            
        except Exception as e:
            print(f"Error refining query: {e}")
            return base_query
    
    def validate_sparql_syntax(self, query: str) -> Tuple[bool, str]:
        """Basic validation of SPARQL query syntax"""
        required_keywords = ['SELECT', 'WHERE']
        
        query_upper = query.upper()
        
        for keyword in required_keywords:
            if keyword not in query_upper:
                return False, f"Missing required keyword: {keyword}"
        
        # Check for balanced braces
        open_braces = query.count('{')
        close_braces = query.count('}')
        
        if open_braces != close_braces:
            return False, f"Unbalanced braces: {open_braces} open, {close_braces} close"
        
        # Check for basic SPARQL structure
        if 'SELECT' in query_upper and 'WHERE' in query_upper:
            select_pos = query_upper.find('SELECT')
            where_pos = query_upper.find('WHERE')
            if select_pos > where_pos:
                return False, "SELECT clause should come before WHERE clause"
        
        return True, "Query syntax appears valid"
    
    def generate_verification_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate specific verification queries for different types of claims"""
        queries = []
        
        # Analyze claim type
        claim_lower = claim.lower()
        
        if any(word in claim_lower for word in ['born', 'birth', 'birthday']):
            queries.extend(self._generate_birth_queries(claim, entities))
        
        if any(word in claim_lower for word in ['died', 'death', 'died']):
            queries.extend(self._generate_death_queries(claim, entities))
        
        if any(word in claim_lower for word in ['founded', 'established', 'created']):
            queries.extend(self._generate_founding_queries(claim, entities))
        
        if any(word in claim_lower for word in ['president', 'ceo', 'leader', 'director']):
            queries.extend(self._generate_position_queries(claim, entities))
        
        if any(word in claim_lower for word in ['located', 'in', 'capital']):
            queries.extend(self._generate_location_queries(claim, entities))
        
        # If no specific type detected, generate general queries
        if not queries:
            queries = self.generate_sparql_query(claim, entities)
        
        return queries
    
    def _generate_birth_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate queries for birth-related claims"""
        queries = []
        
        for entity in entities:
            if entity.label == 'PERSON':
                query = f"""
{self.prefixes}

SELECT ?person ?personLabel ?birthDate ?birthPlace ?birthPlaceLabel WHERE {{
  ?person rdfs:label "{entity.text}"@en .
  OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
  OPTIONAL {{ ?person wdt:P19 ?birthPlace . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
                sparql_query = SPARQLQuery(
                    query=query,
                    entities=[entity.text],
                    relations=['P569', 'P19'],  # birth date, place of birth
                    query_type='property',
                    confidence=0.9,
                    description=f"Birth information for {entity.text}"
                )
                queries.append(sparql_query)
        
        return queries
    
    def _generate_death_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate queries for death-related claims"""
        queries = []
        
        for entity in entities:
            if entity.label == 'PERSON':
                query = f"""
{self.prefixes}

SELECT ?person ?personLabel ?deathDate ?deathPlace ?deathPlaceLabel WHERE {{
  ?person rdfs:label "{entity.text}"@en .
  OPTIONAL {{ ?person wdt:P570 ?deathDate . }}
  OPTIONAL {{ ?person wdt:P20 ?deathPlace . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
                sparql_query = SPARQLQuery(
                    query=query,
                    entities=[entity.text],
                    relations=['P570', 'P20'],  # death date, place of death
                    query_type='property',
                    confidence=0.9,
                    description=f"Death information for {entity.text}"
                )
                queries.append(sparql_query)
        
        return queries
    
    def _generate_founding_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate queries for founding/establishment claims"""
        queries = []
        
        for entity in entities:
            if entity.label in ['ORG', 'WORK_OF_ART']:
                query = f"""
{self.prefixes}

SELECT ?item ?itemLabel ?inception ?founder ?founderLabel WHERE {{
  ?item rdfs:label "{entity.text}"@en .
  OPTIONAL {{ ?item wdt:P571 ?inception . }}
  OPTIONAL {{ ?item wdt:P112 ?founder . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
                sparql_query = SPARQLQuery(
                    query=query,
                    entities=[entity.text],
                    relations=['P571', 'P112'],  # inception, founder
                    query_type='property',
                    confidence=0.9,
                    description=f"Founding information for {entity.text}"
                )
                queries.append(sparql_query)
        
        return queries
    
    def _generate_position_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate queries for position/role claims"""
        queries = []
        
        for entity in entities:
            if entity.label == 'PERSON':
                query = f"""
{self.prefixes}

SELECT ?person ?personLabel ?position ?positionLabel ?startTime ?endTime WHERE {{
  ?person rdfs:label "{entity.text}"@en .
  ?person wdt:P39 ?position .
  OPTIONAL {{ ?person p:P39/pq:P580 ?startTime . }}
  OPTIONAL {{ ?person p:P39/pq:P582 ?endTime . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
                sparql_query = SPARQLQuery(
                    query=query,
                    entities=[entity.text],
                    relations=['P39', 'P580', 'P582'],  # position held, start time, end time
                    query_type='property',
                    confidence=0.8,
                    description=f"Position information for {entity.text}"
                )
                queries.append(sparql_query)
        
        return queries
    
    def _generate_location_queries(self, claim: str, entities: List[Entity]) -> List[SPARQLQuery]:
        """Generate queries for location-related claims"""
        queries = []
        
        for entity in entities:
            if entity.label in ['GPE', 'ORG', 'PERSON']:
                query = f"""
{self.prefixes}

SELECT ?item ?itemLabel ?location ?locationLabel ?country ?countryLabel WHERE {{
  ?item rdfs:label "{entity.text}"@en .
  OPTIONAL {{ ?item wdt:P131 ?location . }}
  OPTIONAL {{ ?item wdt:P17 ?country . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
                sparql_query = SPARQLQuery(
                    query=query,
                    entities=[entity.text],
                    relations=['P131', 'P17'],  # located in, country
                    query_type='property',
                    confidence=0.8,
                    description=f"Location information for {entity.text}"
                )
                queries.append(sparql_query)
        
        return queries

def demo_sparql_generation():
    """Demo function to test SPARQL query generation"""
    generator = SPARQLGenerator()
    
    # Mock entities for demo
    entities = [
        Entity("Barack Obama", 0, 12, "PERSON", 0.9),
        Entity("Hawaii", 25, 31, "GPE", 0.8),
        Entity("1961", 35, 39, "DATE", 0.7)
    ]
    
    claim = "Barack Obama was born in Hawaii in 1961."
    
    print(f"Claim: {claim}")
    print(f"Entities: {[e.text for e in entities]}")
    print("\nGenerated SPARQL queries:")
    
    queries = generator.generate_verification_queries(claim, entities)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i} ({query.query_type}):")
        print(f"Description: {query.description}")
        print(f"Confidence: {query.confidence}")
        print(f"Query:\n{query.query}")
        
        # Validate query
        is_valid, message = generator.validate_sparql_syntax(query.query)
        print(f"Validation: {'✓' if is_valid else '✗'} {message}")

if __name__ == "__main__":
    demo_sparql_generation()