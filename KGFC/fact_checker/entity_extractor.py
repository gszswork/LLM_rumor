"""
Entity Extraction Module for Fact-Checking System
This module extracts topic entities from claims using various NLP techniques.
"""

import re
import json
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import spacy
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Entity:
    """Represents an extracted entity with its metadata"""
    text: str
    start: int
    end: int
    label: str
    confidence: float = 1.0
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

class EntityExtractor:
    """Extracts entities from claims using multiple approaches"""
    
    def __init__(self, use_gemini: bool = True, use_spacy: bool = True):
        self.use_gemini = use_gemini
        self.use_spacy = use_spacy
        
        # Initialize Gemini
        if self.use_gemini:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            else:
                print("Warning: GEMINI_API_KEY not found in environment")
                self.use_gemini = False
        
        # Initialize spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Initialize sentence transformer for similarity
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def extract_entities_gemini(self, claim: str) -> List[Entity]:
        """Extract entities using Gemini API"""
        if not self.use_gemini:
            return []
        
        prompt = f"""
        Extract all named entities from the following claim. Focus on entities that can be found in knowledge graphs like Wikidata.
        
        Return the result as a JSON list where each entity has:
        - "text": the entity text as it appears in the claim
        - "start": character start position
        - "end": character end position  
        - "label": entity type (PERSON, ORG, GPE, EVENT, WORK_OF_ART, etc.)
        - "aliases": list of alternative names/forms
        
        Claim: "{claim}"
        
        JSON:
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
                entities = []
                for ent_data in entities_data:
                    entity = Entity(
                        text=ent_data.get('text', ''),
                        start=ent_data.get('start', 0),
                        end=ent_data.get('end', 0),
                        label=ent_data.get('label', 'UNKNOWN'),
                        confidence=0.9,
                        aliases=ent_data.get('aliases', [])
                    )
                    entities.append(entity)
                return entities
        except Exception as e:
            print(f"Error with Gemini entity extraction: {e}")
        
        return []
    
    def extract_entities_spacy(self, claim: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        if not self.use_spacy:
            return []
        
        doc = self.nlp(claim)
        entities = []
        
        for ent in doc.ents:
            # Filter for entity types likely to be in knowledge graphs
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 
                             'LAW', 'LANGUAGE', 'NORP', 'FAC', 'PRODUCT']:
                entity = Entity(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    confidence=0.8
                )
                entities.append(entity)
        
        return entities
    
    def extract_entities_pattern(self, claim: str) -> List[Entity]:
        """Extract entities using pattern matching for common cases"""
        entities = []
        
        # Pattern for years
        year_pattern = r'\b(19|20)\d{2}\b'
        for match in re.finditer(year_pattern, claim):
            entity = Entity(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                label='DATE',
                confidence=0.7
            )
            entities.append(entity)
        
        # Pattern for numbers that might be quantities
        number_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|percent|%)\b'
        for match in re.finditer(number_pattern, claim, re.IGNORECASE):
            entity = Entity(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                label='QUANTITY',
                confidence=0.6
            )
            entities.append(entity)
        
        # Pattern for quoted text (often titles or names)
        quote_pattern = r'"([^"]+)"'
        for match in re.finditer(quote_pattern, claim):
            entity = Entity(
                text=match.group(1),
                start=match.start(1),
                end=match.end(1),
                label='WORK_OF_ART',
                confidence=0.5
            )
            entities.append(entity)
        
        return entities
    
    def merge_entities(self, entity_lists: List[List[Entity]]) -> List[Entity]:
        """Merge entities from different extractors, removing duplicates"""
        all_entities = []
        for entity_list in entity_lists:
            all_entities.extend(entity_list)
        
        # Sort by start position
        all_entities.sort(key=lambda x: x.start)
        
        # Remove overlapping entities, keeping the one with higher confidence
        merged = []
        for entity in all_entities:
            should_add = True
            for existing in merged:
                # Check for overlap
                if (entity.start < existing.end and entity.end > existing.start):
                    # There's overlap
                    if entity.confidence <= existing.confidence:
                        should_add = False
                        break
                    else:
                        # Remove the existing entity with lower confidence
                        merged.remove(existing)
                        break
            
            if should_add:
                merged.append(entity)
        
        return merged
    
    def filter_entities(self, entities: List[Entity], claim: str) -> List[Entity]:
        """Filter entities based on relevance and quality"""
        filtered = []
        
        for entity in entities:
            # Skip very short entities
            if len(entity.text.strip()) < 2:
                continue
            
            # Skip common stopwords
            if entity.text.lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at']:
                continue
            
            # Skip pure numbers unless they're dates or specific quantities
            if entity.text.isdigit() and entity.label not in ['DATE', 'QUANTITY']:
                continue
            
            filtered.append(entity)
        
        return filtered
    
    def extract_entities(self, claim: str) -> List[Entity]:
        """Main method to extract entities from a claim using all available methods"""
        entity_lists = []
        
        # Extract using different methods
        if self.use_gemini:
            gemini_entities = self.extract_entities_gemini(claim)
            entity_lists.append(gemini_entities)
        
        if self.use_spacy:
            spacy_entities = self.extract_entities_spacy(claim)
            entity_lists.append(spacy_entities)
        
        pattern_entities = self.extract_entities_pattern(claim)
        entity_lists.append(pattern_entities)
        
        # Merge and filter entities
        merged_entities = self.merge_entities(entity_lists)
        filtered_entities = self.filter_entities(merged_entities, claim)
        
        return filtered_entities
    
    def get_entity_context(self, claim: str, entity: Entity, window: int = 50) -> str:
        """Get surrounding context for an entity"""
        start = max(0, entity.start - window)
        end = min(len(claim), entity.end + window)
        return claim[start:end]

def demo_entity_extraction():
    """Demo function to test entity extraction"""
    extractor = EntityExtractor()
    
    test_claims = [
        "Barack Obama was born in Hawaii in 1961.",
        "The COVID-19 pandemic started in 2019 and affected millions worldwide.",
        "Apple Inc. released the iPhone in 2007.",
        "The movie 'Titanic' won 11 Academy Awards in 1998.",
        "Einstein's theory of relativity was published in 1915."
    ]
    
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        entities = extractor.extract_entities(claim)
        print("Extracted entities:")
        for entity in entities:
            print(f"  - {entity.text} ({entity.label}) [{entity.start}:{entity.end}] confidence: {entity.confidence}")

if __name__ == "__main__":
    demo_entity_extraction()