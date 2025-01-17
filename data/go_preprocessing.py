from goatools.obo_parser import *
import time
from data.dag_analysis import *
from data.ProxyTerm import ProxyTerm


def create_dag(file, rel=False):
    """Returns a DAG from an .obo file, with additional root of all namespaces."""
    if rel:
        go_dag = GODag(file, optional_attrs={"relationship"})
    else:
        go_dag = GODag(file)

    # Add a root above the three namespace roots, for easy DAG traversal
    namespace_roots = ("GO:0008150", "GO:0003674", "GO:0005575")
    super_root = GOTerm()
    super_root.item_id = "GO:0000000"
    super_root.name = "GO root"
    super_root.namespace = ""
    super_root.level = -1
    super_root.depth = -1
    super_root.children = {go_dag[namespace_root] for namespace_root in namespace_roots}
    for namespace_root in namespace_roots:
        go_dag[namespace_root].parents.add(super_root)
    if rel:
        super_root.relationship = dict()
        super_root.relationship_rev = dict()

    go_dag["GO:0000000"] = super_root
    return go_dag


def copy_dag(dag):
    """Copy the GO DAG to prevent rebuilding from scratch (to save time)."""
    new_dag = dict()
    rel_check = True
    rel = False
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

        # Check if relationships are enabled
        if rel_check:
            if hasattr(dag[term_id], "relationship"):
                rel = True
            rel_check = False

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

    # Set additional relationships
    if rel:
        for term_id in dag:
            rel = dag[term_id].relationship
            rel_rev = dag[term_id].relationship_rev
            new_dag[term_id].relationship = {}
            new_dag[term_id].relationship_rev = {}
            for rel_type, partners in rel.items():
                new_dag[term_id].relationship[rel_type] = {new_dag[partner.item_id] for partner in partners}
            for rel_type, partners in rel_rev.items():
                new_dag[term_id].relationship_rev[rel_type] = {new_dag[partner.item_id] for partner in partners}

    return new_dag


def filter_by_namespace(go, namespaces):
    """Returns a DAG with only terms from the given namespace(s). (Always includes namespace root.)"""
    filtered_go = copy_dag(go)
    for term_id in go.keys():
        if not (go[term_id].namespace in namespaces):
            # Keep namespace root but remove unwanted child namespaces
            if go[term_id].namespace == "":
                filtered_go[term_id].children = {child for child in go[term_id].children if
                                                 child.namespace in namespaces}
            # Remove unwanted terms and their references
            else:
                term_to_delete = filtered_go.pop(term_id)
                for child in term_to_delete.children:
                    child.parents.discard(term_to_delete)
                for parent in term_to_delete.parents:
                    parent.children.discard(term_to_delete)
                # If present, remove relationship references as well
                if hasattr(term_to_delete, "relationship"):
                    for rel_type, partners in term_to_delete.relationship.items():
                        for partner in partners:
                            partner.relationship_rev[rel_type].discard(term_to_delete)
                    for rel_type, partners in term_to_delete.relationship_rev.items():
                        for partner in partners:
                            partner.relationship[rel_type].discard(term_to_delete)
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
    """If a node A has parents B and C, and B is also an (indirect) parent of C, remove edge AB."""
    pruning_events = 0
    # Define direct and indirect parent sets
    for term_id in go.keys():
        direct_parent_ids = {parent.item_id for parent in go[term_id].parents}
        indirect_parent_ids = set()
        for parent_id in direct_parent_ids:
            indirect_parent_ids.update(go[parent_id].get_all_parents())

        # Check for skip condition
        for parent_id in direct_parent_ids:
            if parent_id in indirect_parent_ids:
                go[term_id].parents.remove(go[parent_id])
                go[parent_id].children.remove(go[term_id])
                pruning_events += 1

                # Update level and depth after skip removal
                update_level_and_depth(go[term_id])

    print(f"Pruning events: {pruning_events}")
    return pruning_events


def merge_chains(go: dict[str, GOTerm], threshold_parents=1, threshold_children=1):
    """If a non-leaf node A has #children <= n and #parents <= m, remove node A.
    Parent(s) of A become(s) the new parent(s) of A's children."""
    merge_events = 0
    merged_term_ids = []
    term_ids = go.keys()
    # Debug:
    removed_terms = []
    for term_id in term_ids:
        term = go[term_id]

        # Skip over alternative IDs, they will be removed together with their corresponding term
        if is_alternative_id(go, term_id):
            continue

        parents = term.parents
        children = term.children
        # Check merge conditions (root and leaves are never merged)
        if (len(children) > 0) and (len(parents) > 0):
            if (len(parents) <= threshold_parents) & (len(children) <= threshold_children):
                # Debug:
                removed_terms.append(term_id)
                # Update parent-child relations
                for parent in parents:
                    parent.children.remove(term)
                    parent.children.update(children)
                for child in children:
                    child.parents.remove(term)
                    child.parents.update(parents)

                    # Update level and depth recursively for all descendants
                    update_level_and_depth(child)

                # Keep track of removed terms and their pseudonyms
                merged_term_ids.append(term_id)
                merged_term_ids += term.alt_ids
                merge_events += 1

    # Remove merged terms from DAG
    for merged_id in merged_term_ids:
        go.pop(merged_id)

    print(f"Merge events: {merge_events}")
    return merge_events, removed_terms


def merge_prune_until_convergence(go: dict[str, GOTerm], threshold_parents=1, threshold_children=1):
    """Iteratively apply merging and pruning for the given merge conditions, until the DAG converges."""
    print("\n----- START: Merge-Prune until convergence -----")
    pruning_events, merge_events = 1, 1
    original_go_size = len(go)
    # Debug:
    removed_terms = []
    while merge_events > 0:
        merge_events, removed = merge_chains(go, threshold_parents, threshold_children)
        pruning_events = prune_skip_connections(go)
        removed_terms.append(removed)
    print(f"Remaining nodes: {len(go)}/{original_go_size}")
    print("----- COMPLETED: Merge-Prune until convergence -----")
    return removed_terms


def update_level_and_depth(term: GOTerm):
    """Set level and depth of the given node and update all its descendants."""
    if len(term.parents) == 0:
        if term.item_id != "GO:0000000":
            print("WARNING: Current term has no parents, but is not original root.")

    else:
        term.level = min(parent.level for parent in term.parents) + 1
        term.depth = max(parent.depth for parent in term.parents) + 1
    for child in term.children:
        update_level_and_depth(child)


def insert_proxy_terms(go: dict[str, GOTerm], root, original_dag_size):
    """From the given root, recursively traverse the graph until an imbalanced child in found. If the current node
    is on the shorter branch above the child, insert a ProxyTerm between parent and child to increase the branch length.
    Depending on the complexity of the graph, multiple passes might be needed to remove all imbalances."""
    if len(root.children) == 0:
        return

    # Check for imbalanced children
    imbalanced_children = set()
    for child in root.children:
        if is_imbalanced(child):
            # Check if parent is on the shorter branch of the imbalance
            if child.level == root.level + 1:
                imbalanced_children.add(child)

    if len(imbalanced_children) > 0:
        # If-statement used for correctly naming the new balancing proxy term
        if isinstance(root, ProxyTerm):
            proxy_item_id = "Proxy:" + str(len(go) - original_dag_size + 1) + "_" + root.item_id[7 + len(
                str(len(go) - original_dag_size)):]
        else:
            proxy_item_id = "Proxy:" + str(len(go) - original_dag_size + 1) + "_" + root.item_id

        # Place proxy between root and all properly imbalanced children
        go[proxy_item_id] = ProxyTerm(proxy_item_id, {root}, imbalanced_children)
        update_level_and_depth(go[proxy_item_id])

    # Once balanced, move down to the children of the children, until leaf layer
    for child in root.children:
        if not is_imbalanced(child):
            insert_proxy_terms(go, child, original_dag_size)


def balance_until_convergence(go: dict[str, GOTerm], root_id="GO:0000000"):
    """Iteratively apply balancing to the given DAG, until all nodes are balanced."""
    print("\n----- START: Balancing DAG with proxies -----")
    original_size = len(go)
    imbalanced = sum(is_imbalanced(term) for term in go.values())
    dag_size = original_size
    stall = False
    stall_counter = 5
    while imbalanced > 0:
        insert_proxy_terms(go, go[root_id], original_size)
        imbalanced = sum(is_imbalanced(term) for term in go.values())

        if not stall:
            # print(f"Proxy terms added: {len(go) - dag_size}")
            # print(f"Imbalanced terms left: {imbalanced}")
            pass

        # Check if the loop stalls, and terminate if needed
        if (len(go) - dag_size == 0) and (imbalanced > 0):
            if stall_counter == 0:
                break

            print(f"WARNING: Imbalance can not be resolved.\nLoop will be terminated in {stall_counter}...")
            stall = True
            stall_counter -= 1
        dag_size = len(go)
    print(f"Number of inserted balancing proxies: {len(go) - original_size}")
    print("----- COMPLETED: Balancing DAG with proxies -----")


def pull_leaves_down(go: dict[str, GOTerm], original_dag_size):
    """Add proxies above leaves until all leaves have equal depth."""
    print("\n----- START: Pull leaves to maximum depth -----")
    pre_proxy_size = len(go)
    leaf_ids = all_leaf_ids(go)
    max_depth = max(go[leaf_id].depth for leaf_id in leaf_ids)
    for leaf_id in leaf_ids:
        while go[leaf_id].depth < max_depth:
            proxy_item_id = "Proxy:" + str(len(go) - original_dag_size + 1) + "_" + leaf_id
            go[proxy_item_id] = ProxyTerm(proxy_item_id, {p for p in go[leaf_id].parents}, {go[leaf_id]})
            update_level_and_depth(go[proxy_item_id])

    print(f"Number of inserted leaf proxies: {len(go) - pre_proxy_size}")
    print("----- COMPLETED: Pull leaves to maximum depth -----")


def relationships_to_parents(go_rel: dict[str, GOTerm]):
    """Update parent-child relationships to include all possible relationships."""
    relationship_check = True
    for term_id in go_rel.keys():
        term = go_rel[term_id]
        if relationship_check:
            if not hasattr(term, "relationship"):
                raise Exception("GO terms must have an attribute 'relationship'.")
            relationship_check = False

        for relationship_type, members in term.relationship.items():
            term.parents.update(members)
            for member in members:
                member.children.add(term)
    update_level_and_depth(go_rel["GO:0000000"])


if __name__ == "__main__":
    t_start = time.time()
    obo_file = "go-basic.obo"
    use_reference = 0
    if use_reference:
        # Original GO DAG metrics for comparison
        t_ref_start = time.time()
        go_complete_ref = create_dag(obo_file)  # 44017 terms
        go_bp_ref = filter_by_namespace(go_complete_ref, "biological_process")  # 28539 terms
        dag_layers_ref = layers_with_duplicates(go_bp_ref)
        overlap_ref = layer_overlap(dag_layers_ref)
        t_ref_end = time.time()
        print(
            f"Reference processing time: {(t_ref_end - t_ref_start) // 60:.0f}m {(t_ref_end - t_ref_start) % 60:.0f}s")

    # Full-scale GO DAG, GO-BP, greedy layerization and layer overlap
    go_complete = create_dag(obo_file)
    go_bp = filter_by_namespace(go_complete, "biological_process")
    dag_layers = layers_with_duplicates(go_bp)
    overlap = layer_overlap(dag_layers)

    # Converged DAG
    converged_layers = layers_with_duplicates(go_bp)
    converged_overlap = layer_overlap(converged_layers)

    t_end = time.time()

    print(f"Total runtime: {(t_end - t_start) // 60:.0f}m {(t_end - t_start) % 60:.0f}s")
