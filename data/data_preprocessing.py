import pandas as pd
import time


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
    matches = gene_set1.intersection(gene_set2)
    with open(path, "w") as f:
        f.write("\n".join(list(matches)))


if __name__ == '__main__':
    data = pd.read_csv("../../GO_TCGA/GE.csv.gz", usecols=[1], compression='gzip')
    name_2_id = read_gene_names_to_uniprot_ids(path="../../GO_TCGA/gene_name2uniprotkb.tsv")
    names, ids = read_gene_names_and_ids(path="../../GO_TCGA/gene_name2uniprotkb.tsv")
    go_ids = read_gene_ids("../../GO_TCGA/gene_ids_in_go_bp.txt")
    save_gene_matches(set(go_ids), set(ids), path="../../GO_TCGA/gene_matches_bp.txt")

    print(f"Data Genes: {len(name_2_id.keys())}")
    print(f"  Data IDs: {len(ids)}")
    print(f"    GO IDs: {len(go_ids)}")

    t = time.time()
    id_matches = set()
    for gene_id in ids:
        if gene_id in go_ids:
            id_matches.add(gene_id)
    print(f"ID matches: {len(id_matches)}")
    print(f"{time.time() - t:.2f} seconds")
    pass
