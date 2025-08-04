"""
WikiData Query Execution Module for Fact-Checking System
This module executes SPARQL queries against WikiData and processes the results.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
import pandas as pd
from urllib.parse import quote
from sparql_generator import SPARQLQuery

@dataclass
class QueryResult:
    """Represents the result of a SPARQL query execution"""
    query: SPARQLQuery
    success: bool
    results: List[Dict[str, Any]]
    execution_time: float
    error_message: Optional[str] = None
    result_count: int = 0
    
    def __post_init__(self):
        if self.success:
            self.result_count = len(self.results)

class WikiDataClient:
    """Client for executing SPARQL queries against WikiData"""
    
    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql"):
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # User agent for requests
        self.user_agent = "FactChecker/1.0 (https://github.com/your-org/fact-checker)"
        self.sparql.addCustomHttpHeader("User-Agent", self.user_agent)
    
    def execute_query(self, query: SPARQLQuery, timeout: int = 30) -> QueryResult:
        """Execute a SPARQL query and return results"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        start_time = time.time()
        
        try:
            self.sparql.setQuery(query.query)
            self.sparql.setTimeout(timeout)
            
            results = self.sparql.query().convert()
            
            execution_time = time.time() - start_time
            self.last_request_time = time.time()
            
            # Process results
            processed_results = self._process_sparql_results(results)
            
            return QueryResult(
                query=query,
                success=True,
                results=processed_results,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.last_request_time = time.time()
            
            return QueryResult(
                query=query,
                success=False,
                results=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def execute_queries_batch(self, queries: List[SPARQLQuery], 
                            max_concurrent: int = 3) -> List[QueryResult]:
        """Execute multiple queries with rate limiting"""
        results = []
        
        for i, query in enumerate(queries):
            print(f"Executing query {i+1}/{len(queries)}: {query.description}")
            result = self.execute_query(query)
            results.append(result)
            
            if not result.success:
                print(f"Query failed: {result.error_message}")
            else:
                print(f"Query returned {result.result_count} results")
        
        return results
    
    def _process_sparql_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process raw SPARQL results into a cleaner format"""
        if 'results' not in raw_results or 'bindings' not in raw_results['results']:
            return []
        
        processed = []
        for binding in raw_results['results']['bindings']:
            processed_binding = {}
            for var, value_info in binding.items():
                processed_binding[var] = self._extract_value(value_info)
            processed.append(processed_binding)
        
        return processed
    
    def _extract_value(self, value_info: Dict[str, str]) -> str:
        """Extract the actual value from SPARQL result binding"""
        if 'value' in value_info:
            value = value_info['value']
            
            # Clean up WikiData URIs
            if value.startswith('http://www.wikidata.org/entity/'):
                return value.replace('http://www.wikidata.org/entity/', '')
            
            return value
        
        return str(value_info)
    
    def search_entity(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for entities by name using WikiData's search API"""
        search_url = "https://www.wikidata.org/w/api.php"
        
        params = {
            'action': 'wbsearchentities',
            'search': entity_name,
            'language': 'en',
            'format': 'json',
            'limit': limit
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data.get('search', [])
            
        except Exception as e:
            print(f"Error searching for entity '{entity_name}': {e}")
            return []
    
    def get_entity_info(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wikibase: <http://www.wikidata.org/ontology#>
        
        SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
          wd:{entity_id} ?property ?value .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 100
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            return self._process_sparql_results(results)
        except Exception as e:
            print(f"Error getting entity info for {entity_id}: {e}")
            return {}
    
    def validate_query_results(self, result: QueryResult) -> Dict[str, Any]:
        """Validate and analyze query results"""
        validation = {
            'has_results': result.success and result.result_count > 0,
            'result_count': result.result_count,
            'execution_time': result.execution_time,
            'query_type': result.query.query_type,
            'confidence': result.query.confidence
        }
        
        if result.success and result.results:
            # Analyze result structure
            first_result = result.results[0]
            validation['variables'] = list(first_result.keys())
            validation['has_labels'] = any('Label' in var for var in first_result.keys())
            
            # Check for common properties
            common_props = ['birthDate', 'deathDate', 'inception', 'location', 'position']
            validation['found_properties'] = [prop for prop in common_props 
                                            if any(prop in var for var in first_result.keys())]
        
        return validation
    
    def format_results_for_reasoning(self, results: List[QueryResult]) -> List[Dict[str, Any]]:
        """Format query results for use in ToG reasoning"""
        formatted_triplets = []
        
        for result in results:
            if not result.success or not result.results:
                continue
            
            for row in result.results:
                # Convert query results to triplet format
                for var_name, value in row.items():
                    if var_name.endswith('Label'):
                        continue  # Skip label variables
                    
                    # Create subject-predicate-object triplets
                    subject = None
                    predicate = var_name
                    object_val = value
                    
                    # Try to find subject from entity names in the query
                    for entity in result.query.entities:
                        if entity in str(row.values()):
                            subject = entity
                            break
                    
                    if subject:
                        triplet = {
                            'subject': subject,
                            'predicate': predicate,
                            'object': object_val,
                            'source': 'wikidata',
                            'confidence': result.query.confidence,
                            'query_type': result.query.query_type
                        }
                        formatted_triplets.append(triplet)
        
        return formatted_triplets
    
    def get_property_info(self, property_id: str) -> Dict[str, Any]:
        """Get information about a WikiData property"""
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?propertyLabel ?description WHERE {{
          wd:{property_id} rdfs:label ?propertyLabel .
          OPTIONAL {{ wd:{property_id} schema:description ?description . }}
          FILTER(LANG(?propertyLabel) = "en")
          FILTER(LANG(?description) = "en")
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            processed = self._process_sparql_results(results)
            return processed[0] if processed else {}
        except Exception as e:
            print(f"Error getting property info for {property_id}: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test connection to WikiData endpoint"""
        test_query = """
        SELECT ?item WHERE {
          ?item wdt:P31 wd:Q5 .
        }
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(test_query)
            self.sparql.query()
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

def demo_wikidata_client():
    """Demo function to test WikiData client"""
    client = WikiDataClient()
    
    # Test connection
    print("Testing connection to WikiData...")
    if client.test_connection():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return
    
    # Search for entities
    print("\nSearching for 'Barack Obama'...")
    entities = client.search_entity("Barack Obama", limit=3)
    for entity in entities[:3]:
        print(f"  - {entity.get('label', 'N/A')} ({entity.get('id', 'N/A')}): {entity.get('description', 'N/A')}")
    
    # Test a simple query
    if entities:
        entity_id = entities[0]['id']
        print(f"\nGetting info for entity {entity_id}...")
        info = client.get_entity_info(entity_id)
        print(f"Found {len(info)} properties")
        for item in info[:5]:  # Show first 5 properties
            prop = item.get('property', 'N/A')
            prop_label = item.get('propertyLabel', 'N/A')
            value = item.get('value', 'N/A')
            value_label = item.get('valueLabel', 'N/A')
            print(f"  - {prop_label} ({prop}): {value_label} ({value})")

if __name__ == "__main__":
    demo_wikidata_client()