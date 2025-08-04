# AI-Powered Fact-Checking System using ToG

This fact-checking system implements the workflow you specified:
1. **Entity Extraction**: Extract topic entities from claims
2. **SPARQL Generation**: Generate SPARQL queries using Gemini API
3. **WikiData Querying**: Execute queries against WikiData
4. **ToG Reasoning**: Use Think-on-Graph reasoning to verify claims

## Architecture

The system consists of four main components:

- **EntityExtractor**: Extracts entities from claims using Gemini API, spaCy, and pattern matching
- **SPARQLGenerator**: Generates targeted SPARQL queries for WikiData using Gemini API
- **WikiDataClient**: Executes SPARQL queries against WikiData and processes results
- **ToGReasoner**: Performs multi-step reasoning on knowledge graphs to verify claims

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

3. Set up environment variables:
```bash
# Create .env file
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Command Line Interface

**Single claim fact-checking:**
```bash
python fact_checker.py --claim "Barack Obama was born in Hawaii in 1961."
```

**Batch fact-checking from file:**
```bash
python fact_checker.py --file claims.txt --output results.json
```

**Interactive mode:**
```bash
python fact_checker.py
```

**System statistics:**
```bash
python fact_checker.py --stats
```

### Python API

```python
from fact_checker import FactCheckingSystem

# Initialize system
system = FactCheckingSystem()

# Fact-check a single claim
session = system.fact_check("The COVID-19 pandemic started in 2019.")

print(f"Verdict: {session.fact_check_result.verdict}")
print(f"Confidence: {session.fact_check_result.confidence}")
print(f"Explanation: {session.fact_check_result.explanation}")

# Batch processing
claims = [
    "Albert Einstein was born in 1879.",
    "The iPhone was released by Apple in 2007.",
    "Paris is the capital of France."
]

sessions = system.batch_fact_check(claims, "batch_results.json")
```

### Individual Components

**Entity Extraction:**
```python
from entity_extractor import EntityExtractor

extractor = EntityExtractor()
entities = extractor.extract_entities("Barack Obama was born in Hawaii in 1961.")

for entity in entities:
    print(f"{entity.text} ({entity.label}) - confidence: {entity.confidence}")
```

**SPARQL Generation:**
```python
from sparql_generator import SPARQLGenerator

generator = SPARQLGenerator()
queries = generator.generate_verification_queries(claim, entities)

for query in queries:
    print(f"Query: {query.description}")
    print(f"SPARQL: {query.query}")
```

**WikiData Querying:**
```python
from wikidata_client import WikiDataClient

client = WikiDataClient()
results = client.execute_queries_batch(queries)

for result in results:
    if result.success:
        print(f"Found {result.result_count} results")
```

## Configuration

Create a `config.json` file to customize system behavior:

```json
{
  "use_gemini_for_entities": true,
  "use_spacy": true,
  "confidence_threshold": 0.7,
  "evidence_threshold": 2,
  "max_reasoning_depth": 3
}
```

## Output Format

The system generates detailed JSON output for each fact-checking session:

```json
{
  "claim": "Barack Obama was born in Hawaii in 1961.",
  "session_id": "fc_1703123456",
  "timestamp": 1703123456.789,
  "entities": [
    {
      "text": "Barack Obama",
      "start": 0,
      "end": 12,
      "label": "PERSON",
      "confidence": 0.95
    }
  ],
  "sparql_queries": [...],
  "query_results": [...],
  "fact_check_result": {
    "verdict": "SUPPORTED",
    "confidence": 0.92,
    "evidence": [...],
    "reasoning_chain": [...],
    "explanation": "Birth information matches WikiData records"
  },
  "processing_time": 8.45
}
```

## Verdict Types

- **SUPPORTED**: Claim is supported by evidence in WikiData
- **REFUTED**: Claim contradicts evidence in WikiData  
- **INSUFFICIENT_EVIDENCE**: Not enough evidence to make a determination
- **ERROR**: System error during processing

## Features

### Entity Extraction
- Multi-modal approach using Gemini API, spaCy NER, and pattern matching
- Confidence scoring and deduplication
- Support for persons, organizations, locations, dates, and more

### SPARQL Query Generation  
- Intelligent query generation based on claim types
- Template-based fallbacks for reliability
- Query validation and refinement
- Support for birth/death, founding, position, and location queries

### WikiData Integration
- Rate-limited querying to respect WikiData's terms
- Robust error handling and retry logic
- Result processing and knowledge graph extraction
- Entity search and property lookup

### ToG Reasoning
- Multi-step reasoning chain (entity verification, property checking, temporal reasoning, consistency analysis)
- Confidence aggregation across reasoning steps
- Evidence collection and explanation generation
- Adaptive verdict determination

## Limitations

- Requires Gemini API access for optimal performance
- Dependent on WikiData availability and coverage
- May struggle with very recent events not yet in WikiData
- Complex claims requiring multi-hop reasoning may need refinement

## Testing

Run the demo functions to test individual components:

```bash
python entity_extractor.py
python sparql_generator.py  
python wikidata_client.py
python tog_reasoner.py
```

## Contributing

1. Ensure all components pass their individual tests
2. Test with diverse claim types and edge cases
3. Follow the existing code structure and documentation patterns
4. Consider rate limiting and ethical use of external APIs

## License

This project follows the same license as the original ToG implementation.