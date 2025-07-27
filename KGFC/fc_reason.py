import os, csv, json

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

        datas.append(cur_data_dict)    
    

    with open(dataset_path + "/fc_"+datasetname + ".csv", "w", encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)
    return datas






def fc_reason(claim, doc_kg):
    # KGQA style fact-checking. claim does not have KG.

    pass

def fc_reason_double_kg(claim, doc_kg, claim_kg): 
    pass


if __name__ == "__main__":
    construct_kg("AggreFact-CNN")

    print(datasetname_list[0].lower())