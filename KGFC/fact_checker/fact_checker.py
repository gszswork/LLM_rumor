"""
Main Fact-Checking System Orchestrator
This module coordinates all components to perform end-to-end fact-checking.
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path

# Import all components
from entity_extractor import EntityExtractor, Entity
from sparql_generator import SPARQLGenerator, SPARQLQuery
from wikidata_client import WikiDataClient, QueryResult
from tog_reasoner import ToGReasoner, FactCheckResult

@dataclass
class FactCheckingSession:
    """Represents a complete fact-checking session"""
    claim: str
    session_id: str
    timestamp: float
    entities: List[Entity]
    sparql_queries: List[SPARQLQuery]
    query_results: List[QueryResult]
    fact_check_result: Optional[FactCheckResult] = None
    processing_time: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'claim': self.claim,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'entities': [asdict(e) for e in self.entities],
            'sparql_queries': [asdict(q) for q in self.sparql_queries],
            'query_results': [asdict(r) for r in self.query_results],
            'fact_check_result': asdict(self.fact_check_result) if self.fact_check_result else None,
            'processing_time': self.processing_time
        }

class FactCheckingSystem:
    """Main orchestrator for the fact-checking system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the fact-checking system"""
        self.config = config or {}
        
        # Initialize components
        print("Initializing fact-checking components...")
        
        try:
            self.entity_extractor = EntityExtractor(
                use_gemini=self.config.get('use_gemini_for_entities', True),
                use_spacy=self.config.get('use_spacy', True)
            )
            print("✓ Entity extractor initialized")
        except Exception as e:
            print(f"✗ Entity extractor failed: {e}")
            self.entity_extractor = None
        
        try:
            self.sparql_generator = SPARQLGenerator()
            print("✓ SPARQL generator initialized")
        except Exception as e:
            print(f"✗ SPARQL generator failed: {e}")
            self.sparql_generator = None
        
        try:
            self.wikidata_client = WikiDataClient()
            if self.wikidata_client.test_connection():
                print("✓ WikiData client initialized and connected")
            else:
                print("⚠ WikiData client initialized but connection test failed")
        except Exception as e:
            print(f"✗ WikiData client failed: {e}")
            self.wikidata_client = None
        
        try:
            self.reasoner = ToGReasoner()
            print("✓ ToG reasoner initialized")
        except Exception as e:
            print(f"✗ ToG reasoner failed: {e}")
            self.reasoner = None
        
        # Validate initialization
        self.is_ready = all([
            self.entity_extractor is not None,
            self.sparql_generator is not None,
            self.wikidata_client is not None,
            self.reasoner is not None
        ])
        
        if self.is_ready:
            print("✓ Fact-checking system ready!")
        else:
            print("⚠ Fact-checking system partially initialized")
    
    def fact_check(self, claim: str, save_session: bool = True) -> FactCheckingSession:
        """Perform complete fact-checking workflow"""
        if not self.is_ready:
            raise RuntimeError("Fact-checking system not properly initialized")
        
        start_time = time.time()
        session_id = f"fc_{int(time.time())}"
        
        print(f"\n{'='*60}")
        print(f"FACT-CHECKING SESSION: {session_id}")
        print(f"Claim: {claim}")
        print(f"{'='*60}")
        
        # Step 1: Extract entities
        print("\n1. EXTRACTING ENTITIES...")
        entities = self.entity_extractor.extract_entities(claim)
        print(f"   Found {len(entities)} entities:")
        for entity in entities:
            print(f"   - {entity.text} ({entity.label}) [confidence: {entity.confidence:.2f}]")
        
        if not entities:
            print("   ⚠ No entities found - creating minimal session")
            session = FactCheckingSession(
                claim=claim,
                session_id=session_id,
                timestamp=start_time,
                entities=[],
                sparql_queries=[],
                query_results=[],
                processing_time=time.time() - start_time
            )
            
            # Create a basic fact check result
            session.fact_check_result = FactCheckResult(
                claim=claim,
                verdict='INSUFFICIENT_EVIDENCE',
                confidence=0.1,
                evidence=[],
                reasoning_chain=[],
                explanation="No entities could be extracted from the claim"
            )
            
            if save_session:
                self._save_session(session)
            return session
        
        # Step 2: Generate SPARQL queries
        print("\n2. GENERATING SPARQL QUERIES...")
        sparql_queries = self.sparql_generator.generate_verification_queries(claim, entities)
        print(f"   Generated {len(sparql_queries)} queries:")
        for i, query in enumerate(sparql_queries, 1):
            print(f"   Query {i}: {query.description} (confidence: {query.confidence:.2f})")
        
        # Step 3: Execute queries
        print("\n3. EXECUTING WIKIDATA QUERIES...")
        query_results = self.wikidata_client.execute_queries_batch(sparql_queries)
        
        successful_results = [r for r in query_results if r.success]
        failed_results = [r for r in query_results if not r.success]
        
        print(f"   Successful queries: {len(successful_results)}")
        print(f"   Failed queries: {len(failed_results)}")
        
        total_results = sum(r.result_count for r in successful_results)
        print(f"   Total knowledge triplets retrieved: {total_results}")
        
        if failed_results:
            print("   Failed query details:")
            for result in failed_results:
                print(f"   - {result.query.description}: {result.error_message}")
        
        # Step 4: Perform reasoning
        print("\n4. PERFORMING TOG REASONING...")
        if successful_results:
            fact_check_result = self.reasoner.fact_check_claim(claim, successful_results)
            print(f"   Verdict: {fact_check_result.verdict}")
            print(f"   Confidence: {fact_check_result.confidence:.2f}")
            print(f"   Evidence pieces: {len(fact_check_result.evidence)}")
        else:
            print("   ⚠ No successful queries - insufficient evidence")
            fact_check_result = FactCheckResult(
                claim=claim,
                verdict='INSUFFICIENT_EVIDENCE',
                confidence=0.2,
                evidence=[],
                reasoning_chain=[],
                explanation="No successful queries were executed to gather evidence"
            )
        
        # Create session
        processing_time = time.time() - start_time
        session = FactCheckingSession(
            claim=claim,
            session_id=session_id,
            timestamp=start_time,
            entities=entities,
            sparql_queries=sparql_queries,
            query_results=query_results,
            fact_check_result=fact_check_result,
            processing_time=processing_time
        )
        
        # Step 5: Save results
        if save_session:
            self._save_session(session)
        
        print(f"\n{'='*60}")
        print(f"FACT-CHECKING COMPLETE")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Final verdict: {fact_check_result.verdict}")
        print(f"Confidence: {fact_check_result.confidence:.2f}")
        print(f"{'='*60}")
        
        return session
    
    def batch_fact_check(self, claims: List[str], output_file: Optional[str] = None) -> List[FactCheckingSession]:
        """Fact-check multiple claims in batch"""
        print(f"Starting batch fact-checking for {len(claims)} claims...")
        
        sessions = []
        for i, claim in enumerate(claims, 1):
            print(f"\nProcessing claim {i}/{len(claims)}")
            try:
                session = self.fact_check(claim, save_session=False)
                sessions.append(session)
            except Exception as e:
                print(f"Error processing claim '{claim}': {e}")
                # Create error session
                error_session = FactCheckingSession(
                    claim=claim,
                    session_id=f"error_{int(time.time())}_{i}",
                    timestamp=time.time(),
                    entities=[],
                    sparql_queries=[],
                    query_results=[],
                    processing_time=0.0
                )
                error_session.fact_check_result = FactCheckResult(
                    claim=claim,
                    verdict='ERROR',
                    confidence=0.0,
                    evidence=[],
                    reasoning_chain=[],
                    explanation=f"Error during processing: {str(e)}"
                )
                sessions.append(error_session)
        
        # Save batch results
        if output_file:
            self._save_batch_results(sessions, output_file)
        
        return sessions
    
    def _save_session(self, session: FactCheckingSession) -> None:
        """Save individual session to file"""
        output_dir = Path("fact_check_results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{session.session_id}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"   Session saved to: {output_file}")
        except Exception as e:
            print(f"   ⚠ Failed to save session: {e}")
    
    def _save_batch_results(self, sessions: List[FactCheckingSession], output_file: str) -> None:
        """Save batch results to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([session.to_dict() for session in sessions], f, indent=2, ensure_ascii=False)
            print(f"Batch results saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save batch results: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and status"""
        return {
            'system_ready': self.is_ready,
            'components': {
                'entity_extractor': self.entity_extractor is not None,
                'sparql_generator': self.sparql_generator is not None,
                'wikidata_client': self.wikidata_client is not None,
                'reasoner': self.reasoner is not None
            },
            'wikidata_connection': self.wikidata_client.test_connection() if self.wikidata_client else False
        }

def main():
    """Command-line interface for the fact-checking system"""
    parser = argparse.ArgumentParser(description="AI-powered Fact-Checking System using ToG")
    parser.add_argument("--claim", type=str, help="Single claim to fact-check")
    parser.add_argument("--file", type=str, help="File containing claims (one per line)")
    parser.add_argument("--output", type=str, help="Output file for batch results")
    parser.add_argument("--config", type=str, help="Configuration file (JSON)")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize system
    system = FactCheckingSystem(config)
    
    # Show stats if requested
    if args.stats:
        stats = system.get_system_stats()
        print("\nSYSTEM STATISTICS:")
        print(json.dumps(stats, indent=2))
        return
    
    # Process single claim
    if args.claim:
        session = system.fact_check(args.claim)
        
        print(f"\nFINAL RESULT:")
        print(f"Claim: {session.claim}")
        print(f"Verdict: {session.fact_check_result.verdict}")
        print(f"Confidence: {session.fact_check_result.confidence:.2f}")
        print(f"Explanation: {session.fact_check_result.explanation}")
        return
    
    # Process file
    if args.file and Path(args.file).exists():
        with open(args.file, 'r', encoding='utf-8') as f:
            claims = [line.strip() for line in f if line.strip()]
        
        output_file = args.output or f"batch_results_{int(time.time())}.json"
        sessions = system.batch_fact_check(claims, output_file)
        
        # Summary statistics
        verdicts = [s.fact_check_result.verdict for s in sessions if s.fact_check_result]
        print(f"\nBATCH SUMMARY:")
        print(f"Total claims: {len(sessions)}")
        for verdict in set(verdicts):
            count = verdicts.count(verdict)
            print(f"{verdict}: {count} ({count/len(verdicts)*100:.1f}%)")
        
        return
    
    # Interactive mode
    print("\nFact-Checking System - Interactive Mode")
    print("Enter claims to fact-check (or 'quit' to exit):")
    
    while True:
        claim = input("\nClaim: ").strip()
        if claim.lower() in ['quit', 'exit', 'q']:
            break
        
        if not claim:
            continue
        
        try:
            session = system.fact_check(claim)
            print(f"\nResult: {session.fact_check_result.verdict}")
            print(f"Confidence: {session.fact_check_result.confidence:.2f}")
            print(f"Explanation: {session.fact_check_result.explanation}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()