"""
Simplified version of ToG reasoning: 

ToG's beam search --> KG ---> LLM reasoning on the KG. 
Since we already use Claude to convert the claim & evidence texts into KGs, we perform the 

'LLM reasoning on two KGs directly.'

e.g. These are the triplets in the claim, these are the triplets in the evidence, is this supported or refuted? 

"""

import json
import time
import os
from fc_helpers import *

def main(datasetname):
    datas = load_datas(datasetname)

    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(timestamp)
    output_filename = f'./{datasetname}_fc_direct_pred_results.json'
    log_filename = f'./{datasetname}_llm_logs_{timestamp}.json'
    
    # Load existing results if file exists
    prediction_results = {}
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                prediction_results = json.load(f)
                print(f"Loaded existing results with {len(prediction_results)} entries")
        except (json.JSONDecodeError, IOError):
            print("Could not load existing results, starting fresh")
            prediction_results = {}

    # for idx, data in enumerate(datas):
    for idx, (i, data) in enumerate(datas.iterrows()):
        # Check if current idx is already processed
        if str(idx) in prediction_results:
            print(f"Skipping idx {idx} - already processed")
            continue
        
        # Start logging for this sample
        start_sample_logging(idx)
        # print(data)
        claim, claim_kg, doc_kg = data['claim_text'], data['claim_kg'], data['doc_kg']
        # claim_topics = [tup[0] for tup in claim_kg] + [tup[2] for tup in claim_kg]
        pred = direct_reason(claim, claim_kg, doc_kg)
        prediction_results[str(idx)] = {'pred': pred, 'claim': claim}
        print(f"idx {idx}: {pred}")
        
        # Save LLM logs for this sample
        save_sample_log(log_filename)
        
        # Write results after every prediction
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(prediction_results, f, ensure_ascii=False, indent=2)


def direct_reason(claim, claim_kg, doc_kg):
    """
    Direct reasoning function.
    """
    claim_chain = []
    for chain in claim_kg:
        triplet_line = f"{chain[0]}, {chain[1]}, {chain[2]}"
        claim_chain.append(triplet_line)

    doc_chain = []
    for chain in doc_kg:
        triplet_line = f"{chain[0]}, {chain[1]}, {chain[2]}"
        doc_chain.append(triplet_line)

    claim_prompt = '\n'.join(claim_chain)
    chain_prompt = '\n'.join(doc_chain)
    
    if not chain_prompt.strip():
        return False, "NOT ENOUGH INFO"
    
    # Use LLM for reasoning (following original ToG approach)
    prompt = double_kg_fact_check_prompt.format(claim_chain, doc_chain)
    
    try:
        response = run_llm(prompt, temperature=0.0, max_tokens=256)
        result = extract_fact_check_answer(response)
        
        if result in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            return True, result
        else:
            return True, "NOT ENOUGH INFO"
            
    except Exception as e:
        # Fallback to heuristic reasoning if LLM fails
        return fallback_reasoning(claim, doc_kg)


if __name__ == "__main__":
    # construct_kg("AggreFact-CNN")
    # main("AggreFact-CNN")
    main("AggreFact-XSum")
