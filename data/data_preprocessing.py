import numpy as np
import pandas as pd


def read_gene_names_to_uniprot_ids(path: str):
    with open(path, "r") as f:
        data_pairs = [pair.split() for pair in f.read().splitlines()]
        data_dict = dict()
        for name, uniprot_id in data_pairs:
            if name in data_dict.keys():
                data_dict[name].append(uniprot_id)
            else:
                data_dict[name] = [uniprot_id]
    return data_dict


def read_uniprot_ids_to_gene_names(path: str):
    with open(path, "r") as f:
        data_pairs = [pair.split() for pair in f.read().splitlines()]
        data_dict = dict()
        for name, uniprot_id in data_pairs:
            if uniprot_id in data_dict.keys():
                data_dict[uniprot_id].append(name)
            else:
                data_dict[uniprot_id] = [name]
    return data_dict


def read_gene_names_and_ids(path: str):
    with open(path, "r") as f:
        data_names_ids = f.read().split()
        data_names = [data_names_ids[i] for i in range(0, len(data_names_ids), 2)]
        data_ids = [data_names_ids[i] for i in range(1, len(data_names_ids), 2)]
        return data_names, data_ids


def read_gene_ids(path: str):
    with open(path, "r") as f:
        gene_ids_from_file = f.read().split()
    return gene_ids_from_file


def save_gene_matches(gene_set1: set[str], gene_set2: set[str], path: str):
    matches = sorted(gene_set1.intersection(gene_set2))
    with open(path, "w") as f:
        f.write("\n".join(list(matches)))


def save_list(a, path: str):
    with open(path, "w") as f:
        f.write("\n".join(a))


if __name__ == '__main__':
    write_to_file = False
    if write_to_file:
        data = pd.read_csv("../../GO_TCGA/GE.csv.gz", usecols=[1], compression='gzip')
        # Retrieve all genes in data as gene name and UniProt ID
        name_2_id = read_gene_names_to_uniprot_ids(path="../../GO_TCGA/gene_data_name2uniprotkb.tsv")
        names, ids = read_gene_names_and_ids(path="../../GO_TCGA/gene_data_name2uniprotkb.tsv")

        # Save list of ID matches between data and GO (no duplicates)
        # Result: different IDs map to the same name
        go_ids = read_gene_ids("../../GO_TCGA/gene_go_bp_id.txt")
        save_gene_matches(set(go_ids), set(ids), path="../../GO_TCGA/gene_matches_bp_id.txt")

        # Save list of gene names corresponding to ID matches
        # Result: duplicate names and duplicate IDs
        match_ids = read_gene_ids("../../GO_TCGA/gene_matches_bp_id.txt")
        match_names = [names[i] for i in range(len(ids)) if ids[i] in match_ids]
        duplicate_match_names = [i for i in set(match_names) if match_names.count(i) > 1]
        save_list(match_names, path="../../GO_TCGA/gene_matches_bp_name.txt")

        # Save (name, ID) pairs for matched IDs
        # Result: duplicate names and duplicate IDs
        match_pairs = [names[i] + "\t" + ids[i] for i in range(len(ids)) if ids[i] in match_ids]
        save_list(match_pairs, "../../GO_TCGA/gene_matches_bp_name2uniprotkb.txt")

        # Greedy approach of matching names to IDs
        # if a name has an ID not yet used -> match name and ID
        # else -> exclude gene
        # Result: 15143 unique names with unique IDs
        data_ids_greedy = []
        data_names_greedy = list(name_2_id.keys())
        id_2_name = read_uniprot_ids_to_gene_names(path="../../GO_TCGA/gene_data_name2uniprotkb.tsv")
        for data_name in data_names_greedy:
            id_in_go = False
            for id_option in name_2_id[data_name]:
                if id_option in go_ids:
                    if id_option not in data_ids_greedy:
                        data_ids_greedy.append(id_option)
                        id_in_go = True
                        break
            if not id_in_go:
                data_ids_greedy.append(None)

            match_pairs = [data_names_greedy[i] + "\t" + data_ids_greedy[i] for i in range(len(data_names_greedy)) if
                           data_ids_greedy[i]]
            save_list(match_pairs, "../../GO_TCGA/gene_matches_bp_name2uniprotkb_greedy.txt")

        # Save dataset containing only gene names with greedy ID (actual IDs not in dataset)
        names, ids = read_gene_names_and_ids(path="../../GO_TCGA/gene_matches_bp_name2uniprotkb_greedy.txt")
        filtered_data = data[data["gene symbol"].isin(names)]
        filtered_data.to_csv("../../GO_TCGA/GE_bp.csv.gz", compression='gzip')

        # Save dataset containing the first 100 genes, including ID
        data = pd.read_csv("../../GO_TCGA/GE_bp.csv.gz", compression='gzip')
        first_hundred_rows = data.head(100)
        first_hundred_rows.insert(3, "gene id", np.array(ids[:100]))
        first_hundred_rows.to_csv("../../GO_TCGA/GE_bp_100.csv.gz", compression='gzip')

        # Test names, ids both for duplicates
        print(len(names))
        print(len(set(names)))
        print(len(ids))
        print(len(set(ids)))
