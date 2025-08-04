"""
Test script for the fact-checking system
Tests individual components and end-to-end functionality.
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_entity_extraction():
    """Test entity extraction component"""
    print("=" * 50)
    print("TESTING ENTITY EXTRACTION")
    print("=" * 50)
    
    try:
        from entity_extractor import EntityExtractor
        
        extractor = EntityExtractor(use_gemini=False, use_spacy=True)  # Use spaCy only for testing
        
        test_claims = [
            "Barack Obama was born in Hawaii in 1961.",
            "Apple Inc. was founded in 1976.",
            "The COVID-19 pandemic started in 2019."
        ]
        
        for claim in test_claims:
            print(f"\nClaim: {claim}")
            entities = extractor.extract_entities(claim)
            print(f"Entities found: {len(entities)}")
            for entity in entities:
                print(f"  - {entity.text} ({entity.label}), confidence: {entity.confidence:.2f}")
        
        print("\n✓ Entity extraction test completed")
        return True
        
    except Exception as e:
        print(f"✗ Entity extraction test failed: {e}")
        return False

def test_sparql_generation():
    """Test SPARQL query generation"""
    print("\n" + "=" * 50)
    print("TESTING SPARQL GENERATION")
    print("=" * 50)
    
    try:
        from sparql_generator import SPARQLGenerator
        from entity_extractor import Entity
        
        generator = SPARQLGenerator()
        
        # Mock entities for testing
        entities = [
            Entity("Barack Obama", 0, 12, "PERSON", 0.9),
            Entity("Hawaii", 25, 31, "GPE", 0.8),
            Entity("1961", 35, 39, "DATE", 0.7)
        ]
        
        claim = "Barack Obama was born in Hawaii in 1961."
        
        print(f"Claim: {claim}")
        print(f"Entities: {[e.text for e in entities]}")
        
        # Test template-based generation (fallback)
        queries = generator.generate_template_queries(claim, entities)
        print(f"\nGenerated {len(queries)} template queries:")
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}:")
            print(f"  Description: {query.description}")
            print(f"  Type: {query.query_type}")
            print(f"  Confidence: {query.confidence}")
            
            # Validate syntax
            is_valid, message = generator.validate_sparql_syntax(query.query)
            print(f"  Validation: {'✓' if is_valid else '✗'} {message}")
        
        print("\n✓ SPARQL generation test completed")
        return True
        
    except Exception as e:
        print(f"✗ SPARQL generation test failed: {e}")
        return False

def test_wikidata_client():
    """Test WikiData client"""
    print("\n" + "=" * 50)
    print("TESTING WIKIDATA CLIENT")
    print("=" * 50)
    
    try:
        from wikidata_client import WikiDataClient
        
        client = WikiDataClient()
        
        # Test connection
        print("Testing connection...")
        if client.test_connection():
            print("✓ Connection successful")
        else:
            print("⚠ Connection test failed")
            return False
        
        # Test entity search
        print("\nTesting entity search...")
        entities = client.search_entity("Barack Obama", limit=3)
        if entities:
            print(f"Found {len(entities)} entities:")
            for entity in entities[:2]:
                print(f"  - {entity.get('label', 'N/A')} ({entity.get('id', 'N/A')})")
        else:
            print("No entities found")
        
        print("\n✓ WikiData client test completed")
        return True
        
    except Exception as e:
        print(f"✗ WikiData client test failed: {e}")
        return False

def test_end_to_end():
    """Test complete fact-checking workflow"""
    print("\n" + "=" * 50)
    print("TESTING END-TO-END WORKFLOW")
    print("=" * 50)
    
    try:
        from fact_checker import FactCheckingSystem
        
        # Use minimal configuration for testing
        config = {
            'use_gemini_for_entities': False,  # Use spaCy only
            'use_spacy': True
        }
        
        system = FactCheckingSystem(config)
        
        if not system.is_ready:
            print("⚠ System not ready - checking component status:")
            stats = system.get_system_stats()
            for component, status in stats['components'].items():
                print(f"  {component}: {'✓' if status else '✗'}")
            return False
        
        # Test with a simple claim
        test_claim = "Paris is the capital of France."
        print(f"\nTesting claim: {test_claim}")
        
        session = system.fact_check(test_claim, save_session=False)
        
        print(f"\nResults:")
        print(f"  Verdict: {session.fact_check_result.verdict}")
        print(f"  Confidence: {session.fact_check_result.confidence:.2f}")
        print(f"  Processing time: {session.processing_time:.2f}s")
        print(f"  Entities found: {len(session.entities)}")
        print(f"  Queries executed: {len(session.query_results)}")
        
        successful_queries = [r for r in session.query_results if r.success]
        print(f"  Successful queries: {len(successful_queries)}")
        
        print("\n✓ End-to-end test completed")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n" + "=" * 50)
    print("TESTING BATCH PROCESSING")
    print("=" * 50)
    
    try:
        from fact_checker import FactCheckingSystem
        
        config = {
            'use_gemini_for_entities': False,  # Use spaCy only for testing
            'use_spacy': True
        }
        
        system = FactCheckingSystem(config)
        
        if not system.is_ready:
            print("⚠ System not ready for batch testing")
            return False
        
        # Test with multiple simple claims
        test_claims = [
            "Paris is the capital of France.",
            "The iPhone was released in 2007.",
            "Albert Einstein was a physicist."
        ]
        
        print(f"Testing {len(test_claims)} claims in batch...")
        
        sessions = system.batch_fact_check(test_claims, output_file=None)
        
        print(f"\nBatch results:")
        for i, session in enumerate(sessions, 1):
            result = session.fact_check_result
            print(f"  Claim {i}: {result.verdict} (confidence: {result.confidence:.2f})")
        
        # Summary statistics
        verdicts = [s.fact_check_result.verdict for s in sessions]
        verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}
        
        print(f"\nSummary:")
        for verdict, count in verdict_counts.items():
            print(f"  {verdict}: {count}")
        
        print("\n✓ Batch processing test completed")
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("FACT-CHECKING SYSTEM TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Entity Extraction", test_entity_extraction),
        ("SPARQL Generation", test_sparql_generation),
        ("WikiData Client", test_wikidata_client),
        ("End-to-End", test_end_to_end),
        ("Batch Processing", test_batch_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠ Some tests failed - check individual results above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)