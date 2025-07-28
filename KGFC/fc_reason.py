import json
import time
from fc_helpers import *

# TODO: run_llm, reasoning, fallbackcall functions. 


def fc_reason(claim, doc_kg, claim_topics):
    """
    Fact-checking reasoning function based on ToG algorithm.
    Follows the iterative deepening structure of main_freebase.py
    
    Args:
        claim (str): The claim text to fact-check
        doc_kg (list): List of RDF triples from document (subject, predicate, object)
        claim_topics (list): List of topic entities from the claim
    
    Returns:
        str: Fact-checking result ("SUPPORTS", "REFUTES", or "NOT ENOUGH INFO")
    """
    
    # Build knowledge graph index from RDF triples
    kg_index = build_kg_index(doc_kg)
    
    # Initialize topic entities similar to ToG
    topic_entity = {entity: entity for entity in claim_topics if entity in kg_index['all_entities']} 
    
    if not topic_entity:
        return "NOT ENOUGH INFO"
    
    # ToG-like iterative deepening search
    cluster_chain_of_entities = []
    pre_relations = []
    pre_heads = [-1] * len(topic_entity)
    max_depth = 3
    width = 3
    
    for depth in range(1, max_depth + 1):
        current_entity_relations_list = []
        
        # Relation search and pruning for each topic entity
        i = 0
        for entity_id, entity_name in topic_entity.items():
            if entity_id != "[FINISH_ID]":
                # Extend from the current entity
                relations_with_scores = relation_search_prune_kg(
                    entity_id, entity_name, pre_relations, 
                    pre_heads[i] if i < len(pre_heads) else -1, 
                    claim, kg_index, width
                )
                current_entity_relations_list.extend(relations_with_scores)
            i += 1
        

        # Entity search and scoring
        total_candidates = []
        total_scores = []
        total_relations = []
        total_entities_id = []
        total_topic_entities = []
        total_head = []

        for entity_rel in current_entity_relations_list:
            # Search for entities connected by this relation
            if entity_rel['head']:
                entity_candidates_id = entity_search_kg(entity_rel['entity'], entity_rel['relation'], True, kg_index)
            else:
                entity_candidates_id = entity_search_kg(entity_rel['entity'], entity_rel['relation'], False, kg_index)

            if len(entity_candidates_id) == 0:
                continue
                
            #print(entity_rel['head'] ,entity_candidates_id)
            # Score entity candidates
            scores, entity_candidates, entity_candidates_id = entity_score_kg(
                claim, entity_candidates_id, entity_rel['score'], entity_rel['relation']
            )
            
            # Update history
            # total_candidates is not needed. We do not want get the entity as the answer. 
            total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history_kg(
                entity_candidates, entity_rel, scores, entity_candidates_id, 
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head
            )
        
            # print(total_candidates)
        if len(total_candidates) == 0:
            break
        
        # Entity pruning
        flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune_kg(
            total_entities_id, total_relations, total_candidates, total_topic_entities, 
            total_head, total_scores, width
        )
        
        cluster_chain_of_entities.append(chain_of_entities)
        
        if flag:
            # Reasoning step - check if current evidence is sufficient for fact-checking
            stop, result = reasoning(claim, cluster_chain_of_entities)
            if stop:
                return result
            else:
                # Continue to next depth with new topic entities
                topic_entity = {entity: entity for entity in entities_id}
                continue
        else:
            break
    
    # Final reasoning with all collected evidence
    if cluster_chain_of_entities:
        _, result = fallback_reasoning(claim, cluster_chain_of_entities)
        return result
    else:
        return "NOT ENOUGH INFO"


def fc_reason_double_kg(claim, doc_kg, claim_kg): 
    # My KG-wise matching fact-checking algorithm. 
    pass


def main(datasetname):
    datas = load_datas(datasetname)

    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(timestamp)
    output_filename = f'./{datasetname}_pred_results_{timestamp}.json'
    prediction_results = {}

    for idx, data in enumerate(datas):
        claim, claim_kg, doc_kg = data['claim_text'], data['claim_kg'], data['doc_kg']
        claim_topics = [tup[0] for tup in claim_kg] + [tup[2] for tup in claim_kg]
        pred = fc_reason(claim, doc_kg, claim_topics)
        prediction_results[idx] = {'pred': pred, 'claim': claim}  # <-- Uncommented
        print(pred)
        # Write results after every prediction
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(prediction_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # construct_kg("AggreFact-CNN")
    main("AggreFact-CNN")
