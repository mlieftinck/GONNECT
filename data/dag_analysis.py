from goatools.obo_parser import GOTerm
import matplotlib.pyplot as plt
import numpy as np

from data.ProxyTerm import ProxyTerm


def is_alternative_id(go: dict[str, GOTerm], term_id):
    """Returns True if the given ID is a pseudonym of a GO term object"""
    return term_id != go[term_id].item_id


def is_imbalanced(term: GOTerm) -> bool:
    """Returns True if the given GO term is imbalanced"""
    return term.level != term.depth


def all_leaf_ids(go: dict[str, GOTerm]):
    """Returns all term IDs with 0 children."""
    leaves = set()
    for term in go:
        if len(go[term].children) == 0:
            leaves.add(term)
    return leaves


def direct_parents(go: dict[str, GOTerm], term_ids):
    """Returns a set of all the immediate parents from the IDs in the input set."""
    parent_set = set()
    for term_id in term_ids:
        term = go[term_id]
        parents = set([parent.item_id for parent in term.parents])
        parent_set = parent_set.union(parents)
    return parent_set


def layers_with_duplicates(go: dict[str, GOTerm], only_id=False):
    """Returns a list of sets containing leaves and consecutive immediate parents.
    Terms can appear multiple times in different layers. Argument determines whether
    layers consist of objects (GOTerm) or IDs (String)."""
    layers = []
    leaves = all_leaf_ids(go)

    if only_id:
        layers.append(leaves)
    else:
        layers.append({go[term_id] for term_id in leaves})

    parents = direct_parents(go, leaves)
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


def plot_depth_distribution(go: dict[str, GOTerm], term_ids, sub_fig=None, alpha=0.5, bins=np.arange(18) - 0.5,
                            title="Distribution of GO-term depths", show_proxy=False):
    depths = []
    max_depth_term_id = "GO:0000001"
    min_depth_term_id = "GO:0000001"
    for term_id in term_ids:
        if not show_proxy:
            if term_id[:5] == "Proxy":
                continue
        depths.append(go[term_id].depth)
        if go[term_id].depth > go[max_depth_term_id].depth:
            max_depth_term_id = term_id
        if go[term_id].depth < go[min_depth_term_id].depth:
            min_depth_term_id = term_id

    # print(f"max leaf depth: {max(depths)} ({max_depth_term_id})")
    # print(f"min leaf depth: {min(depths)} ({min_depth_term_id})")
    if sub_fig:
        sub_fig.set_xlabel("Depth")
        sub_fig.set_ylabel("Number of terms")
        sub_fig.set_xticks(np.arange(stop=18, step=2))
        sub_fig.set_title(title)
        sub_fig.hist(depths, alpha=alpha, bins=bins, edgecolor="k")
    else:
        plt.xlabel("Depth")
        plt.ylabel("Number of terms")
        plt.xticks(np.arange(stop=18, step=2))
        plt.title(title)
        plt.hist(depths, alpha=alpha, bins=bins, edgecolor="k")


def print_dag_info(dag: dict[str, GOTerm]):
    proxies = len([term for term in dag.values() if isinstance(term, ProxyTerm)])
    terms = len(dag.values())
    print(f"Number of nodes: {terms}")
    print(f"Number of leafs: {len(all_leaf_ids(dag))}")
    print(f"Proxy terms: {proxies}/{terms} = {proxies / terms * 100:.1f}%")
    print(f"Max depth: {max(term.depth for term in dag.values())}")
