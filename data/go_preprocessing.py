from goatools.obo_parser import *
import time
from dag_analysis import *
from data.ProxyTerm import ProxyTerm


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

                    # Update level and depth recursively for all descendants
                    update_level_and_depth(go, child)

                # Keep track of removed terms and their pseudonyms
                merged_term_ids.append(term_id)
                merged_term_ids += term.alt_ids
                merge_events += 1

    # Remove merged terms from DAG
    for merged_id in merged_term_ids:
        go.pop(merged_id)

    print(f"Total amount of merge events: {merge_events}")
    return merge_events


def update_level_and_depth(go: dict[str, GOTerm], term: GOTerm):
    """Set level and depth of argument and update all descendants."""
    term.level = min(parent.level for parent in term.parents) + 1
    term.depth = max(parent.depth for parent in term.parents) + 1
    for child in term.children:
        update_level_and_depth(go, child)


def insert_proxy_terms(go: dict[str, GOTerm], root, original_dag_size):
    """From the given root, recursively traverse the graph until an imbalance in found. If the current root
    is on the shorter branch of the imbalance, insert a ProxyTerm to increase the branch length. Depending
    on the complexity of the graph, multiple passes might be needed to remove all imbalances."""

    if len(root.children) == 0:
        return

    imbalanced_children = set()
    for child in root.children:
        if is_imbalanced(child):
            # Check if this is the shorter branch of the imbalance
            if child.level == root.level + 1:
                imbalanced_children.add(child)

    if len(imbalanced_children) > 0:
        if isinstance(root, ProxyTerm):
            proxy_item_id = "Proxy:" + str(len(go) - original_dag_size + 1) + "_" + root.item_id[-10:]
        else:
            proxy_item_id = "Proxy:" + str(len(go) - original_dag_size + 1) + "_" + root.item_id
        go[proxy_item_id] = ProxyTerm(proxy_item_id, {root}, imbalanced_children)
        update_level_and_depth(go, go[proxy_item_id])
        # print(f"{proxy_item_id} inserted below {root.item_id}")

    for child in root.children:
        if not is_imbalanced(child):
            insert_proxy_terms(go, child, original_dag_size)


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
