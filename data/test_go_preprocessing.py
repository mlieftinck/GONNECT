from unittest import TestCase
from DAGGenerator import DAGGenerator
import matplotlib.pyplot as plt
from go_preprocessing import create_dag, copy_dag, filter_go_by_namespace, layers_with_duplicates, layer_overlap, \
    prune_skip_connections, merge_chains, all_leafs, plot_depth_distribution
from ProxyTerm import ProxyTerm

go_bp = filter_go_by_namespace(create_dag("go-basic.obo"), "biological_process")


class Test(TestCase):

    def test_copy_dag_small(self):
        dag = DAGGenerator.dag_update_rule1()
        dag_copy = copy_dag(dag)
        dag.pop("D")
        dag["B"].children.add(dag["A"])
        self.assertNotEqual(len(dag_copy["B"].children), len(dag["B"].children))

    def test_copy_dag(self):
        go_copy = copy_dag(go_bp)
        original_kv_pairs = [[key, go_bp[key]] for key in go_bp.keys()]
        copied_kv_pairs = [[key, go_copy[key]] for key in go_copy.keys()]

        for i in range(len(original_kv_pairs)):
            self.assertEqual(original_kv_pairs[i][0], copied_kv_pairs[i][0])
            self.assertEqual(original_kv_pairs[i][1].item_id, copied_kv_pairs[i][1].item_id)
            self.assertEqual(original_kv_pairs[i][1].level, copied_kv_pairs[i][1].level)
            self.assertEqual(original_kv_pairs[i][1].depth, copied_kv_pairs[i][1].depth)
            self.assertEqual(len(original_kv_pairs[i][1].parents), len(copied_kv_pairs[i][1].parents))

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    def test_prune_skip_connections_small(self):
        dag = DAGGenerator.dag_update_rule1()
        greedy_layers = layers_with_duplicates(dag)
        overlap = layer_overlap(greedy_layers)
        overlapping_terms = sum([len(overlap[pair]) for pair in overlap])
        self.assertEqual(overlapping_terms, 1)
        prune_skip_connections(dag)
        greedy_layers_after_pruning = layers_with_duplicates(dag)
        overlap_after_pruning = layer_overlap(greedy_layers_after_pruning)
        overlapping_terms_after_pruning = sum([len(overlap_after_pruning[pair]) for pair in overlap_after_pruning])
        self.assertEqual(overlapping_terms_after_pruning, 0)
        self.assertNotEqual(overlapping_terms, overlapping_terms_after_pruning)

    def test_prune_skip_connections_go(self):
        go = copy_dag(go_bp)
        intersections_a = layer_overlap(layers_with_duplicates(go))
        pruning_events = prune_skip_connections(go)
        intersections_b = layer_overlap(layers_with_duplicates(go))
        sizes_a = [len(intersections_a[i]) for i in intersections_a]
        sizes_b = [len(intersections_b[i]) for i in intersections_b]
        self.assertEqual(pruning_events, 0)

    #      A
    #    / |
    #  /   B
    # | /
    # C
    def test_merge_chains_small(self):
        dag = DAGGenerator.dag_update_rule2()
        term_before_merge = len(dag.keys())
        merge_events = merge_chains(dag)
        terms_after_merge = len(dag.keys())
        self.assertEqual(merge_events, 1)
        self.assertLess(terms_after_merge, term_before_merge)

    def test_merge_chains_go(self):
        dag = copy_dag(go_bp)
        terms_before_merge = len(dag.keys())
        merge_events = merge_chains(dag)
        terms_after_merge = len(dag.keys())
        self.assertGreater(merge_events, 0)
        self.assertLess(terms_after_merge, terms_before_merge)

    def test_single_merge_effect_on_overlap(self):
        # dag = filter_go_by_namespace(create_dag("go-basic.obo"), "biological_process")
        dag = copy_dag(go_bp)
        layers = layers_with_duplicates(dag)
        overlap = layer_overlap(layers)
        merge_chains(dag, threshold_parents=1, threshold_children=1)
        merged_layers = layers_with_duplicates(dag)
        merged_overlap = layer_overlap(merged_layers)

        overlap_difference = {}
        for key in overlap.keys():
            overlap_difference[key] = len(merged_overlap[key]) - len(overlap[key])

    def test_merge_chains_until_convergence(self):
        dag = copy_dag(go_bp)
        # Plot leaf depth distribution before and after merge-prune convergence
        leaf_ids = all_leafs(dag)
        plot_depth_distribution(dag, leaf_ids)
        # First merge, then prune:
        pruning_events = 1
        merge_events = 1
        parent_threshold = 1
        children_threshold = 1
        while pruning_events + merge_events > 0:
            merge_events = merge_chains(dag, parent_threshold, children_threshold)
            pruning_events = prune_skip_connections(dag)
        leaf_ids_post = all_leafs(dag)
        plot_depth_distribution(dag, leaf_ids_post)
        plt.title(f"Distribution of leaf depth after merge-pruning \n"
                  f"(#parents = {parent_threshold}, #children = {children_threshold})")
        plt.show()

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    def test_set_level_and_depth_small(self):
        dag = DAGGenerator.dag_update_rule1()
        dag["E"] = ProxyTerm("E", {dag["B"]}, {dag["D"]})
        self.assertEqual(dag["E"].depth, dag["C"].depth)
        self.assertEqual(dag["E"].level, dag["C"].level + 1)


