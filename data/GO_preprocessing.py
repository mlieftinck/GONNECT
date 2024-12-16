import numpy as np
from goatools.obo_parser import *
import json
import time
import DAGGenerator
import matplotlib.pyplot as plt


def store_dag(dag, output_file):
    with open(output_file, "w") as location:
        json.dump(dag, location)


def load_dag(file):
    return json.load(file)


def create_dag(file):
    """Returns a DAG from an .obo file."""
    return GODag(file)


def copy_dag(dag):
    """Copy the GO DAG to prevent rebuilding from scratch (saving time)"""
    new_dag = {}
    # Create object per term
    for term_id in dag:
        term = dag[term_id]
        new_term = GOTerm()
        new_term.item_id = term.item_id
        new_term.name = term.name
        new_term.namespace = term.namespace
        new_term.level = term.level
        new_term.depth = term.depth
        new_term.is_obsolete = term.is_obsolete
        new_term.alt_ids = term.alt_ids
        new_dag[term_id] = new_term

    # Link alternative IDs to single object
    for term_id in dag:
        if is_alternative_id(dag, term_id):
            new_dag[term_id] = new_dag[dag[term_id].item_id]

    # Set parent-child relations
    for term_id in dag:
        children = dag[term_id].children
        parents = dag[term_id].parents
        for child in children:
            new_dag[term_id].children.add(new_dag[child.item_id])
        for parent in parents:
            new_dag[term_id].parents.add(new_dag[parent.item_id])

    return new_dag


def filter_go_by_namespace(go, namespace):
    """Returns a DAG with only terms from the given namespace."""
    filtered_go = {}
    for term_id in go:
        if go[term_id].namespace == namespace:
            filtered_go[term_id] = go[term_id]
    return filtered_go


def all_leafs(go: dict[str, GOTerm]):
    """Returns all terms with 0 children."""
    leafs = set()
    for term in go:
        if len(go[term].get_all_children()) == 0:
            leafs.add(term)
    return leafs


def direct_parents(go: dict[str, GOTerm], term_ids):
    """Returns a set of all the immediate parents from the IDs in the input set."""
    parent_set = set()
    for term_id in term_ids:
        term = go[term_id]
        parents = set([parent.item_id for parent in term.parents])
        parent_set = parent_set.union(parents)
    return parent_set


def layers_with_duplicates(go: dict[str, GOTerm], only_id=False):
    """Returns a list of sets containing leafs and consecutive immediate parents.
    Terms can appear multiple times in different layers. Argument determines whether
    layers consist of objects (GOTerm) or IDs (String)."""
    layers = []
    leafs = all_leafs(go)

    if only_id:
        layers.append(leafs)
    else:
        layers.append({go[term_id] for term_id in leafs})

    parents = direct_parents(go, leafs)
    while len(parents) > 0:
        if only_id:
            layers.append(parents)
        else:
            layers.append({go[term_id] for term_id in parents})
        parents = direct_parents(go, parents)
    return layers


def number_of_children(go: dict[str, GOTerm], term_ids):
    """Returns a dictionary where the terms in the input set
    are sorted by total number of children (recursively)."""
    terms_by_number_of_children = {}
    for term_id in term_ids:
        children = go[term_id].get_all_children()
        if len(children) in terms_by_number_of_children:
            terms_by_number_of_children[len(children)] = terms_by_number_of_children[len(children)].union([go[term_id]])
        else:
            terms_by_number_of_children[len(children)] = {go[term_id]}
    return terms_by_number_of_children


def layer_overlap(layers):
    """Returns a dict containing the intersection of each possible pair of layers"""
    overlap = {}
    k = 0
    for k in range(len(layers) - 1):
        for i in range(k + 1, len(layers)):
            if i == k:
                pass
            else:
                overlap[(k, i)] = layers[k].intersection(layers[i])
    return overlap


# 28-11-2024
# NOTE: Merging overlap between layers is unproductive, since this removes high-order terms
#       that occur often, but at different depths. Merging should be more vertical than on
#       a per-layer basis. Also, this method currently does not remove terms from the DAG.
def merge_overlap(go: dict[str, GOTerm], layer1: int, layer2: int):
    """Removes every term in the intersection of layer1 and layer2,
    transferring all children to parents and all parents to children."""
    layers = layers_with_duplicates(go)
    overlap_dict = layer_overlap((layers[layer1], layers[layer2]))
    overlapping_set = overlap_dict[(0, 1)]
    for term in overlapping_set:
        term.namespace += " REMOVED"
        parents = term.parents
        children = term.children
        for child in children:
            child.parents.remove(term)
            child.parents = child.parents.union(parents)

        for parent in parents:
            parent.children.remove(term)
            parent.children = parent.children.union(children)


def is_alternative_id(go: dict[str, GOTerm], term_id):
    """Returns True if the given ID is a pseudonym of a GO term object"""
    return term_id != go[term_id].item_id


def prune_skip_connections(go: dict[str, GOTerm]):
    """If a node A has parents B and C, and B is also a parent of C, remove edge AB."""
    pruning_events = 0
    for term_id in go:
        direct_parent_ids = {parent.item_id for parent in go[term_id].parents}
        indirect_parent_ids = set()
        for parent_id in direct_parent_ids:
            indirect_parent_ids.update(go[parent_id].get_all_parents())

        for parent_id in direct_parent_ids:
            if parent_id in indirect_parent_ids:
                go[term_id].parents.remove(go[parent_id])
                go[parent_id].children.remove(go[term_id])
                pruning_events += 1
    print(f"Total amount of pruning events: {pruning_events}")
    return pruning_events


def merge_chains(go: dict[str, GOTerm], threshold_parents=1, threshold_children=1):
    """If a node A has #children <= n and #parents <= m, remove node A.
    Parent(s) of A become(s) the new parent(s) of A's children."""
    merge_events = 0
    merged_term_ids = []
    term_ids = go.keys()
    for term_id in term_ids:
        term = go[term_id]

        # Skip over alternative IDs, they will be removed together with their corresponding term
        if is_alternative_id(go, term_id):
            continue

        # Check merge conditions
        parents = term.parents
        children = term.children
        if len(children) > 0:
            if (len(parents) <= threshold_parents) & (len(children) <= threshold_children):

                # Update parent-child relations
                for parent in parents:
                    parent.children.remove(term)
                    parent.children.update(children)
                for child in children:
                    child.parents.remove(term)
                    child.parents.update(parents)

                    # Update depths
                    child.depth -= 1
                    childs_children = child.get_all_children()
                    for childs_child_id in childs_children:
                        go[childs_child_id].depth -= 1

                # Keep track of removed terms and their pseudonyms
                merged_term_ids.append(term_id)
                merged_term_ids += term.alt_ids
                merge_events += 1

    # Remove merged terms from DAG
    for merged_id in merged_term_ids:
        go.pop(merged_id)

    print(f"Total amount of merge events: {merge_events}")
    return merge_events


def plot_depth_distribution(go: dict[str, GOTerm], term_ids, alpha=0.5, bins=np.arange(18) - 0.5,
                            title="Distribution of GO-term depths"):
    depths = []
    for term_id in term_ids:
        depths.append(go[term_id].depth)

    plt.xlabel("GO-DAG depth")
    plt.ylabel("Number of GO-terms")
    plt.xticks(np.arange(stop=18, step=2))
    plt.title(title)
    plt.hist(depths, alpha=alpha, bins=bins, edgecolor="k")


if __name__ == "__main__":
    t_start = time.time()
    obo_file = "go-basic.obo"
    use_reference = 0
    if use_reference:
        # Original GO DAG metrics for comparison
        t_ref_start = time.time()
        go_complete_ref = create_dag(obo_file)  # 44017 terms
        go_bp_ref = filter_go_by_namespace(go_complete_ref, "biological_process")  # 28539 terms
        dag_layers_ref = layers_with_duplicates(go_bp_ref)
        overlap_ref = layer_overlap(dag_layers_ref)
        t_ref_end = time.time()
        print(
            f"Reference processing time: {(t_ref_end - t_ref_start) // 60:.0f}m {(t_ref_end - t_ref_start) % 60:.0f}s")

    # Full-scale GO DAG, GO-BP, greedy layerization and layer overlap
    go_complete = create_dag(obo_file)
    go_bp = filter_go_by_namespace(go_complete, "biological_process")
    dag_layers = layers_with_duplicates(go_bp)
    overlap = layer_overlap(dag_layers)

    # Converged DAG
    converged_layers = layers_with_duplicates(go_bp)
    converged_overlap = layer_overlap(converged_layers)

    t_end = time.time()

    print(f"Total runtime: {(t_end - t_start) // 60:.0f}m {(t_end - t_start) % 60:.0f}s")
