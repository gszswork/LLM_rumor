import os, csv, json
import pandas as pd
import re
import time
import hashlib
import pickle
from pathlib import Path
from openai import OpenAI
import google.generativeai as genai
import dotenv

# Load environment variables from a .env file if present
dotenv.load_dotenv()

# Cache configuration
CACHE_DIR = Path("llm_cache")
CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"

# Global logging for prompt/response pairs per data sample
_current_sample_log = {}
_current_sample_id = None

def ensure_cache_dir():
    """Ensure cache directory exists"""
    if CACHE_ENABLED:
        CACHE_DIR.mkdir(exist_ok=True)

def start_sample_logging(sample_id):
    """Start logging for a new data sample"""
    global _current_sample_log, _current_sample_id
    _current_sample_id = sample_id
    _current_sample_log = {}

def log_llm_call(prompt, response, call_type="llm_call"):
    """Log a prompt/response pair for the current sample"""
    global _current_sample_log
    if _current_sample_id is not None:
        call_count = len(_current_sample_log)
        _current_sample_log[str(call_count)] = {
            "type": call_type,
            "prompt": prompt,
            "response": response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def get_sample_log():
    """Get the current sample's log"""
    return _current_sample_log.copy()

def save_sample_log(log_file_path):
    """Save the current sample log to file"""
    global _current_sample_log, _current_sample_id
    if _current_sample_id is not None and _current_sample_log:
        # Load existing logs
        all_logs = {}
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    all_logs = json.load(f)
            except (json.JSONDecodeError, IOError):
                all_logs = {}
        
        # Add current sample log
        all_logs[str(_current_sample_id)] = _current_sample_log
        
        # Save updated logs
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=2)

def generate_cache_key(prompt, temperature, max_tokens, engine):
    """Generate unique cache key for LLM call parameters"""
    cache_data = f"{prompt}|{temperature}|{max_tokens}|{engine}"
    return hashlib.sha256(cache_data.encode()).hexdigest()

def get_cached_response(cache_key):
    """Get cached response if exists"""
    if not CACHE_ENABLED:
        return None
    
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Cache read error: {e}")
    return None

def save_to_cache(cache_key, response):
    """Save response to cache"""
    if not CACHE_ENABLED:
        return
    
    ensure_cache_dir()
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)
    except Exception as e:
        print(f"Cache write error: {e}")

def run_llm(prompt, temperature=0.0, max_tokens=256, api_key="", engine="gemini-2.0-flash"):
    """
    Real LLM function supporting OpenAI GPT and Google Gemini models with caching
    """
    # Generate cache key
    cache_key = generate_cache_key(prompt, temperature, max_tokens, engine)
    
    # Check cache first
    cached_response = get_cached_response(cache_key)
    if cached_response is not None:
        print(f"Cache hit for {engine}")
        return cached_response
    
    # Cache miss - make actual API call
    print(f"Cache miss for {engine}")
    if engine.lower().startswith('gpt'):
        # Use OpenAI API for GPT models
        response = run_openai_llm(prompt, temperature, max_tokens, api_key, engine)
    elif engine.lower().startswith('gemini'):
        # Use Google Gemini API for Gemini models
        response = run_gemini_llm(prompt, temperature, max_tokens, api_key, engine)
    else:
        print(f"Unsupported engine: {engine}, falling back to mock")
        response = run_llm_mock(prompt, temperature, max_tokens, api_key, engine)
    
    # Log the LLM call for current sample
    log_llm_call(prompt, response, f"llm_call_{engine}")
    
    # Save to cache
    save_to_cache(cache_key, response)
    return response


def run_openai_llm(prompt, temperature, max_tokens, api_key, engine):
    """
    Run LLM using OpenAI API (called from cached run_llm function)
    """
    # Get API key
    api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found, falling back to mock")
        return run_llm_mock(prompt, temperature, max_tokens, api_key, engine)
    
    client = OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt}
    ]
    
    print("start openai API call")
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0
            )
            result = response.choices[0].message.content
            print("end openai API call")
            return result
        except Exception as e:
            print(f"openai error (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)
            else:
                print("Max retries reached, falling back to mock")
                return run_llm_mock(prompt, temperature, max_tokens, api_key, engine)


def run_gemini_llm(prompt, temperature, max_tokens, api_key, engine):
    """
    Run LLM using Google Gemini API (called from cached run_llm function)
    """
    # Get API key
    api_key = api_key if api_key else os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No Google/Gemini API key found, falling back to mock")
        return run_llm_mock(prompt, temperature, max_tokens, api_key, engine)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # print("start gemini API call")
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Initialize the model
            model = genai.GenerativeModel(engine)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            result = response.text
            print("end gemini API call")
            return result
        except Exception as e:
            print(f"gemini error (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)
            else:
                print("Max retries reached, falling back to mock")
                return run_llm_mock(prompt, temperature, max_tokens, api_key, engine)


def run_llm_mock(prompt, temperature=0.0, max_tokens=256, api_key="", engine="gpt-3.5-turbo"):
    """
    Mock LLM function for fallback when real LLM is unavailable
    """
    prompt_lower = prompt.lower()
    
    # Mock fact-checking reasoning responses
    if "claim:" in prompt_lower and "knowledge triplets:" in prompt_lower:
        if ("works_at" in prompt_lower and "google" in prompt_lower) or \
           ("beat" in prompt_lower and "manchester" in prompt_lower) or \
           ("happy_at" in prompt_lower and "paris" in prompt_lower):
            return "{SUPPORTS}. The evidence clearly supports the claim based on the provided knowledge triplets."
        elif "never" in prompt_lower and ("works" in prompt_lower or "knocked out" in prompt_lower):
            return "{REFUTES}. The evidence contradicts the claim since the triplets show the opposite of what the claim states."
        else:
            return "{NOT ENOUGH INFO}. The evidence is insufficient to fact-check the claim."
    
    # Mock entity scoring
    elif "score the entities" in prompt_lower:
        if "manchester united" in prompt_lower:
            return "0.8, 0.1, 0.05, 0.05"
        else:
            return "0.4, 0.3, 0.3"
    
    # Mock relation extraction - improved to use actual relations from prompt
    elif "retrieve" in prompt_lower and "relations" in prompt_lower:
        # Extract the relations from the prompt - look for LAST "Relations:" line (not the example)
        lines = prompt.split('\n')
        relations_line = None
        for line in reversed(lines):  # Start from the end to get the actual relations, not example
            if line.strip().startswith('Relations:'):
                relations_line = line.replace('Relations:', '').strip()
                break
        
        if relations_line:
            # Parse the actual relations
            actual_relations = [rel.strip() for rel in relations_line.split(';')]
            actual_relations = [rel for rel in actual_relations if rel]  # remove empty
            
            if len(actual_relations) >= 1:
                # Return the first 2-3 relations with scores
                result_lines = []
                total_relations = min(3, len(actual_relations))
                scores = [0.5, 0.3, 0.2]  # descending scores
                
                for i, rel in enumerate(actual_relations[:total_relations]):
                    score = scores[i] if i < len(scores) else 0.1
                    result_lines.append(f"{i+1}. {{{rel} (Score: {score:.1f})}}: Relevant relation for the claim")
                
                return '\n'.join(result_lines)
        
        # Fallback to generic response
        return "1. {is (Score: 0.5)}: General descriptive relation\n2. {has (Score: 0.5)}: General possessive relation"
    
    return "Mock LLM response for fact-checking."

# Fact-checking specific prompts
fact_check_evaluate_prompt = """Given a claim and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to determine if the evidence is sufficient to fact-check the claim and provide a verdict (SUPPORTS, REFUTES, or NOT ENOUGH INFO).

Example 1:
Claim: John works at Google as a software engineer
Knowledge Triplets: John, works_at, Google
John, is, software_engineer
Google, is, technology_company
A: {{SUPPORTS}}. The evidence clearly supports the claim that John works at Google as a software engineer based on the provided knowledge triplets.

Example 2:
Claim: Mary never worked at Microsoft
Knowledge Triplets: Mary, works_at, Microsoft
Mary, is, employee
Microsoft, is, technology_company
A: {{REFUTES}}. The evidence contradicts the claim since the triplets show that Mary works at Microsoft, which refutes the claim that she never worked there.

Example 3:
Claim: The new iPhone costs $1000
Knowledge Triplets: iPhone, manufactured_by, Apple
Apple, is, technology_company
iPhone, is, smartphone
A: {{NOT ENOUGH INFO}}. The evidence does not contain information about the iPhone's price, so it's insufficient to fact-check the claim about the cost.

Claim: {}
Knowledge Triplets: {}
A: """


# Double KG fact-checking prompts 
double_kg_fact_check_prompt = """Given a claim's knowledge graph (KG) triplets and the associated retrieved evidence knowledge graph triplets (entity, relation, entity), you are asked to determine if the evidence is sufficient to fact-check the claim and provide a verdict (SUPPORTS, REFUTES, or NOT ENOUGH INFO) You are strongly encouraged to use as less NOT ENOUGH INFO as possible.

Example 1:
Claim Triplets: John, works_at, Google
John, is, software_engineer
Knowledge Triplets: John, works_at, Google
Google, is, technology_company
A: {{SUPPORTS}}. The evidence clearly supports the claim that John works at Google as a software engineer based on the provided knowledge triplets.

Example 2:
Claim Triplets: Mary, works_at, Google
Knowledge Triplets: Mary, works_at, Microsoft
Mary, is, employee
Microsoft, is, technology_company
A: {{REFUTES}}. The evidence contradicts the claim since the triplets show that Mary works at Microsoft, which refutes the claim that she works at Google. 

Example 3:
Claim Triplets: The new iPhone, costs, $1000
Knowledge Triplets: iPhone, manufactured_by, Apple
Apple, is, technology_company
iPhone, is, smartphone
A: {{NOT ENOUGH INFO}}. The evidence does not contain information about the iPhone's price, so it's insufficient to fact-check the claim about the cost.

Claim Triplets: {}
Knowledge Triplets: {}
A: """

fact_check_relation_extract_prompt = """Please retrieve {} relations (separated by semicolon) that are most relevant for fact-checking the given claim and rate their contribution on a scale from 0 to 1 (the sum of the scores of {} relations is 1).

Example:
Claim: John works at Google
Topic Entity: John
Relations: works_at; is; has; located_in; founded_by; owns; manages
A: 1. {{{{works_at (Score: 0.7)}}}}): This relation is highly relevant as it directly addresses the employment relationship mentioned in the claim.
2. {{{{is (Score: 0.2)}}}}): This relation provides contextual information about John's identity which may be relevant.
3. {{{{has (Score: 0.1)}}}}): This relation might provide additional context about John's attributes.

Claim: {}
Topic Entity: {}
Relations: {}
A: """

fact_check_entity_score_prompt = """Please score the entities' relevance to fact-checking the claim on a scale from 0 to 1 (the sum of the scores of all entities is 1).

Example:
Claim: Chelsea beat Manchester United 1-0
Relation: beat
Entities: Manchester United; Arsenal; Liverpool; Manchester City
Score: 0.8, 0.1, 0.05, 0.05
The entity most relevant to the claim is Manchester United since it's directly mentioned in the claim about Chelsea beating them.

Claim: {}
Relation: {}
Entities: {}
Score: """

datasetname_list = ["AgggreFact-CNN", "AggreFact-XSum", "COVID-Fact"]


def csv_to_rdf_triples(node_csv_path, edge_csv_path):
    # Read nodes: id -> label
    node_dict = {}
    with open(node_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_dict[row['node_id']] = row['node_attr']

    # Read edges and build triples
    rdf_triples = []
    with open(edge_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = node_dict.get(row['src'], row['src'])
            pred = row['edge_attr']
            dst = node_dict.get(row['dst'], row['dst'])
            rdf_triples.append((src, pred, dst))

    return rdf_triples


def construct_kg(datasetname):
    datas = [] # To save all the datas into json.
    dataset_path = f"dataset/extracted_KG/{datasetname}"
    node_path, edge_path  = dataset_path + "/nodes/", dataset_path + "/edges/"
    all_doc_node_files = os.listdir(node_path + "doc/")
    numbers = []
    for f in all_doc_node_files:
        if f.endswith('.csv'):
            try:
                num = int(f[:-4])  # Remove '.csv' and convert to int
                numbers.append(num)
            except ValueError:
                pass 
    indices = max(numbers)

    claim_texts_path = './lightrag_docs/'+ datasetname + '_claims.json'
    with open(claim_texts_path, 'r', encoding='utf-8') as f:
        claim_texts = json.load(f)

    for i in range(indices):

        cur_data_dict = {}
        doc_node_csv_path = node_path + "doc/" + str(i) + ".csv"
        doc_edge_csv_path = edge_path + "doc/" + str(i) + ".csv"
        doc_rdf_triples = csv_to_rdf_triples(doc_node_csv_path, doc_edge_csv_path)

        cur_data_dict['doc_kg'] = doc_rdf_triples

        # print(doc_rdf_triples)

        claim_node_csv_path = node_path + "claim/" + str(i) + ".csv"
        claim_edge_csv_path = edge_path + "claim/" + str(i) + ".csv"
        claim_rdf_triples = csv_to_rdf_triples(claim_node_csv_path, claim_edge_csv_path)
        cur_data_dict['claim_kg'] = claim_rdf_triples
        cur_data_dict['claim_text'] = claim_texts[i]

        datas.append(cur_data_dict)    
    

    with open(dataset_path + "/fc_"+datasetname + ".csv", "w", encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)
    return datas


def load_datas(datasetname):
    dataset_path = f"dataset/extracted_KG/{datasetname}/{datasetname}.pkl"
    # with open(dataset_path, 'r', encoding='utf-8') as f:
    #     datas = json.load(f)
    datas = pd.read_pickle(dataset_path)
    
    # datas is a list of dict, each dict is a data item: 
    # With k-w pairs {'doc_kg', 'claim_kg', 'claim_text'}
    return datas


def build_kg_index(doc_kg):
    """
    Build knowledge graph index from RDF triples for efficient lookup.
    Similar to SPARQL endpoint but using local data.
    """
    # Head relations: entity -> relation -> [objects]
    head_relations = {}
    # Tail relations: entity -> relation -> [subjects] 
    tail_relations = {}
    
    for subject, predicate, obj in doc_kg:
        # Head relations (entity as subject)
        if subject not in head_relations:
            head_relations[subject] = {}
        if predicate not in head_relations[subject]:
            head_relations[subject][predicate] = []
        head_relations[subject][predicate].append(obj)
        
        # Tail relations (entity as object)
        if obj not in tail_relations:
            tail_relations[obj] = {}
        if predicate not in tail_relations[obj]:
            tail_relations[obj][predicate] = []
        tail_relations[obj][predicate].append(subject)
    
    return {'head': head_relations, 'tail': tail_relations, 'all_entities': set([s for s, _, _ in doc_kg] + [o for _, _, o in doc_kg])}


def relation_search_prune_kg(entity_id, entity_name, pre_relations, pre_head, claim, kg_index, width):
    """
    Adapted from relation_search_prune in utils.py for KG-based search.
    Uses LLM to find and score relations for an entity.
    """
    # Get head and tail relations for this entity
    head_relations = list(kg_index['head'].get(entity_id, {}).keys())
    tail_relations = list(kg_index['tail'].get(entity_id, {}).keys())
    
    # Filter out previous relations to avoid cycles
    if len(pre_relations) != 0 and pre_head != -1:
        if pre_head:
            head_relations = [rel for rel in head_relations if rel not in pre_relations]
        else:
            tail_relations = [rel for rel in tail_relations if rel not in pre_relations]
    
    # Combine relations
    all_relations = head_relations + tail_relations
    if not all_relations:
        return []
    
    # Use LLM to score relations (following original ToG approach)
    relations_str = '; '.join(all_relations)
    prompt = fact_check_relation_extract_prompt.format(width, width, claim, entity_name, relations_str)
    
    try:
        result = run_llm(prompt, temperature=0.4, max_tokens=256)
        relations_with_scores = clean_relations_llm(result, entity_id, head_relations)
        if relations_with_scores:
            return relations_with_scores
    except Exception as e:
        print(f"LLM failed with error: {e}")
        pass
    
    # Fallback to text similarity if LLM fails
    relation_scores = []
    for relation in all_relations:
        score = calculate_text_similarity(claim, relation)
        is_head = relation in head_relations
        relation_scores.append({
            'entity': entity_id,
            'relation': relation,
            'score': score,
            'head': is_head
        })
    
    # Sort by score and return top width relations
    relation_scores.sort(key=lambda x: x['score'], reverse=True)
    return relation_scores[:width]


def clean_relations_llm(llm_response, entity_id, head_relations):
    """
    Parse LLM response to extract relations with scores.
    Adapted from clean_relations in utils.py
    """
    pattern = r"{\s*([^{}]+)\s+\(Score:\s+([0-9.]+)\)}"
    relations = []
    
    for match in re.finditer(pattern, llm_response):
        relation = match.group(1).strip()
        score_str = match.group(2)
        
        if ';' in relation:
            continue
            
        try:
            score = float(score_str)
        except ValueError:
            continue
            
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    
    return relations if relations else []


def entity_search_kg(entity_id, relation, head, kg_index):
    """
    Adapted from entity_search in utils.py for KG-based search.
    Find entities connected by a specific relation.
    """
    if head:
        # entity -relation-> ?
        return kg_index['head'].get(entity_id, {}).get(relation, [])
    else:
        # ? -relation-> entity
        return kg_index['tail'].get(entity_id, {}).get(relation, [])


def entity_score_kg(claim, entity_candidates_id, base_score, relation):
    """
    Adapted from entity_score in utils.py for KG-based scoring.
    Uses LLM to score entity candidates based on relevance to claim.
    """
    if not entity_candidates_id:
        return [], [], []
    
    entity_candidates = [str(entity) for entity in entity_candidates_id]
    
    if len(entity_candidates) == 1:
        return [base_score], entity_candidates, entity_candidates_id
    
    # Use LLM to score entities (following original ToG approach)
    entities_str = '; '.join(entity_candidates)
    prompt = fact_check_entity_score_prompt.format(claim, relation, entities_str)
    
    try:
        # TODO: activate this with a true llm call. 
        result = run_llm(prompt, temperature=0.4, max_tokens=256)
        scores = clean_scores_llm(result, entity_candidates)
        scores = [float(s) * base_score for s in scores]
        return scores, entity_candidates, entity_candidates_id
    except:
        print("Not scoring the entity thru llm, fallback and use similarity-based scoring. ")
        pass
    
    # Fallback to text similarity if LLM fails
    scores = []
    for entity in entity_candidates:
        score = calculate_text_similarity(claim, str(entity)) * base_score
        scores.append(score)
    
    return scores, entity_candidates, entity_candidates_id


def clean_scores_llm(llm_response, entity_candidates):
    """
    Parse LLM response to extract entity scores.
    Adapted from clean_scores in utils.py
    """
    # Try to extract comma-separated scores
    scores = re.findall(r'[0-9]+\.?[0-9]*', llm_response)
    scores = [float(s) for s in scores if float(s) <= 1.0]
    
    if len(scores) == len(entity_candidates):
        return scores
    else:
        # Equal distribution fallback
        return [1.0/len(entity_candidates)] * len(entity_candidates)


def update_history_kg(entity_candidates, entity_rel, scores, entity_candidates_id, 
                     total_candidates, total_scores, total_relations, total_entities_id, 
                     total_topic_entities, total_head):
    """
    Adapted from update_history in utils.py.
    Update the search history with new candidates.
    """
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
        scores = [0.0]
    
    candidates_relation = [entity_rel['relation']] * len(entity_candidates)
    topic_entities = [entity_rel['entity']] * len(entity_candidates)
    head_flags = [entity_rel['head']] * len(entity_candidates)
    
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_flags)
    
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def entity_prune_kg(total_entities_id, total_relations, total_candidates, total_topic_entities, 
                   total_head, total_scores, width):
    """
    Adapted from entity_prune in utils.py.
    Prune entities to keep only the top-scored ones.
    """
    if not total_entities_id:
        return False, [], [], [], []
    
    # Combine and sort by scores
    zipped = list(zip(total_entities_id, total_relations, total_candidates, 
                     total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    
    # Keep top width candidates
    top_candidates = sorted_zipped[:width]
    
    # Filter out zero scores
    filtered_candidates = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in top_candidates if score > 0]
    
    if not filtered_candidates:
        return False, [], [], [], []
    
    entities_id, relations, candidates, topics, heads, scores = map(list, zip(*filtered_candidates))
    
    # Build chain of entities (triplets)
    chain_of_entities = [[(topics[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    
    return True, chain_of_entities, entities_id, relations, heads


def reasoning(claim, cluster_chain_of_entities):
    """
    Adapted from reasoning in utils.py for fact-checking.
    Uses LLM to determine if current evidence is sufficient and what the verdict is.
    """
    if not cluster_chain_of_entities:
        return False, "NOT ENOUGH INFO"
    
    # Convert chains to text for LLM analysis (following original ToG format)
    chain_lines = []
    for chain_cluster in cluster_chain_of_entities:
        for chain in chain_cluster:
            if isinstance(chain, (list, tuple)) and len(chain) >= 3:
                triplet_line = f"{chain[0]}, {chain[1]}, {chain[2]}"
                chain_lines.append(triplet_line)
    
    chain_prompt = '\n'.join(chain_lines)
    
    if not chain_prompt.strip():
        return False, "NOT ENOUGH INFO"
    
    # Use LLM for reasoning (following original ToG approach)
    prompt = fact_check_evaluate_prompt.format(claim, chain_prompt)
    
    try:
        response = run_llm(prompt, temperature=0.0, max_tokens=256)
        result = extract_fact_check_answer(response)
        
        if result in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            return True, result
        else:
            return True, "NOT ENOUGH INFO"
            
    except Exception as e:
        # Fallback to heuristic reasoning if LLM fails
        return fallback_reasoning(claim, cluster_chain_of_entities)


def extract_fact_check_answer(text):
    """
    Extract fact-checking answer from LLM response.
    Adapted from extract_answer in utils.py
    """
    # Look for answer in curly braces
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        answer = text[start_index+1:end_index].strip().upper()
        if answer in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            return answer
    
    # Fallback: look for keywords in response
    text_upper = text.upper()
    if "SUPPORTS" in text_upper:
        return "SUPPORTS"
    elif "REFUTES" in text_upper:
        return "REFUTES"
    else:
        return "NOT ENOUGH INFO"


def fallback_reasoning(claim, cluster_chain_of_entities):
    """
    Fallback reasoning using LLM direct answer approach when main LLM fails.
    Follows the generate_without_explored_paths pattern from utils.py
    """
    # Create a direct fact-checking prompt similar to generate_directly
    fallback_prompt = """Given a claim, determine if it should be classified as SUPPORTS, REFUTES, or NOT ENOUGH INFO based on your knowledge.

Claim: The president of the United States in 2021 was Joe Biden.
Answer: Based on factual knowledge, Joe Biden became the 46th president of the United States in January 2021. The answer is {SUPPORTS}.

Claim: The capital of France is London.
Answer: Based on factual knowledge, the capital of France is Paris, not London. The answer is {REFUTES}.

Claim: The population of Mars is exactly 50,000 people.
Answer: Based on current knowledge, there is no established human population on Mars. The answer is {NOT ENOUGH INFO}.

Claim: """ + claim + """
Answer:"""
    
    try:
        # Use run_llm similar to generate_without_explored_paths
        response = run_llm(fallback_prompt, temperature=0.0, max_tokens=256, api_key="")
        
        # Extract the fact-check answer from the response
        result = extract_fact_check_answer(response)
        
        if result in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            return True, result
        else:
            return True, "NOT ENOUGH INFO"  # Conservative fallback
            
    except Exception as e:
        # If even the fallback LLM fails, use simple heuristic
        return simple_heuristic_fallback(claim, cluster_chain_of_entities)


def simple_heuristic_fallback(claim, cluster_chain_of_entities):
    """
    Simple heuristic fallback when all LLM approaches fail.
    """
    # Simple heuristic based on entity matching
    all_triplets = []
    for chain_cluster in cluster_chain_of_entities:
        for chain in chain_cluster:
            all_triplets.extend(chain)
    
    if not all_triplets:
        return False, "NOT ENOUGH INFO"
    
    claim_lower = claim.lower()
    entity_matches = 0
    
    for subject, predicate, obj in all_triplets:
        if (str(subject).lower() in claim_lower or 
            str(obj).lower() in claim_lower):
            entity_matches += 1
    
    if entity_matches > 0:
        return True, "NOT ENOUGH INFO"  # Conservative fallback
    else:
        return False, "NOT ENOUGH INFO"


def calculate_text_similarity(text1, text2):
    """
    Calculate simple text similarity using word overlap.
    Handles underscores and case insensitivity.
    """
    # Normalize text by replacing underscores with spaces and converting to lowercase
    norm_text1 = str(text1).lower().replace('_', ' ')
    norm_text2 = str(text2).lower().replace('_', ' ')
    
    words1 = set(norm_text1.split())
    words2 = set(norm_text2.split())
    
    if not words1 or not words2:
        return 0.1  # Give small default score instead of 0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not intersection:
        return 0.1  # Give small default score for no overlap
    
    return len(intersection) / len(union) if union else 0.1