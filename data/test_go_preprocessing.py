from unittest import TestCase
from goatools.obo_parser import GODag
from DAGGenerator import DAGGenerator
import matplotlib.pyplot as plt
from data.dag_analysis import is_imbalanced, print_dag_info
from data.go_preprocessing import insert_proxy_terms, update_level_and_depth, pull_leaves_down, \
    relationships_to_parents, \
    merge_prune_until_convergence, balance_until_convergence
from go_preprocessing import create_dag, copy_dag, filter_go_by_namespace, layers_with_duplicates, layer_overlap, \
    prune_skip_connections, merge_chains, all_leaf_ids, plot_depth_distribution
from ProxyTerm import ProxyTerm

go_main = create_dag("go-basic.obo")
go_bp_main = filter_go_by_namespace(go_main, "biological_process")
go_rel_main = create_dag("go-basic.obo", rel=True)


class Test(TestCase):

    def test_copy_dag_small(self):
        dag = DAGGenerator.dag1()
        dag_copy = copy_dag(dag)
        dag.pop("D")
        dag["B"].children.add(dag["A"])
        self.assertNotEqual(len(dag_copy["B"].children), len(dag["B"].children))

    def test_copy_dag(self):
        go_copy = copy_dag(go_bp_main)
        original_kv_pairs = [[key, go_bp_main[key]] for key in go_bp_main.keys()]
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
        dag = DAGGenerator.dag1()
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
        go = copy_dag(go_bp_main)
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
        dag = DAGGenerator.dag2()
        term_before_merge = len(dag.keys())
        merge_events = merge_chains(dag)
        terms_after_merge = len(dag.keys())
        self.assertEqual(merge_events, 1)
        self.assertLess(terms_after_merge, term_before_merge)

    def test_merge_chains_go(self):
        dag = copy_dag(go_bp_main)
        terms_before_merge = len(dag.keys())
        merge_events = merge_chains(dag)
        terms_after_merge = len(dag.keys())
        self.assertGreater(merge_events, 0)
        self.assertLess(terms_after_merge, terms_before_merge)

    def test_single_merge_effect_on_overlap(self):
        dag = copy_dag(go_bp_main)
        layers = layers_with_duplicates(dag)
        overlap = layer_overlap(layers)
        merge_chains(dag, threshold_parents=1, threshold_children=1)
        merged_layers = layers_with_duplicates(dag)
        merged_overlap = layer_overlap(merged_layers)

        overlap_difference = {}
        for key in overlap.keys():
            overlap_difference[key] = len(merged_overlap[key]) - len(overlap[key])

    def test_merge_chains_until_convergence(self):
        dag = copy_dag(go_bp_main)
        # Plot leaf depth distribution before and after merge-prune convergence
        leaf_ids = all_leaf_ids(dag)
        plot_depth_distribution(dag, leaf_ids)
        # First merge, then prune:
        pruning_events = 1
        merge_events = 1
        parent_threshold = 1
        children_threshold = 1
        while pruning_events + merge_events > 0:
            merge_events = merge_chains(dag, parent_threshold, children_threshold)
            pruning_events = prune_skip_connections(dag)
        leaf_ids_post = all_leaf_ids(dag)
        plot_depth_distribution(dag, leaf_ids_post)
        plt.title(f"Distribution of leaf depth after merge-pruning \n"
                  f"(#parents = {parent_threshold}, #children = {children_threshold})")
        plt.show()

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    def test_insert_proxy_terms_small(self):
        dag = DAGGenerator.dag1()
        self.assertNotEqual(dag["C"].level, dag["C"].depth)
        insert_proxy_terms(dag, dag["A"], len(dag))
        self.assertEqual(dag["C"].level, dag["C"].depth)

    def test_update_level_and_depth_small(self):
        dag = DAGGenerator.dag1()
        dag["E"] = ProxyTerm("E", {dag["C"]}, set())
        update_level_and_depth(dag["C"])
        self.assertEqual(dag["E"].level, dag["C"].level + 1)
        self.assertEqual(dag["E"].depth, dag["C"].depth + 1)

    def test_insert_proxy_terms_go(self):
        dag = copy_dag(go_bp_main)
        original_size = len(dag)
        go_root = "GO:0008150"
        # Find all imbalanced terms
        imbalanced = sum(is_imbalanced(term) for term in dag.values())
        dag_size = original_size
        print(f"Imbalanced terms: {imbalanced}")
        # Plot baseline
        leaf_ids = all_leaf_ids(dag)
        plot_depth_distribution(dag, leaf_ids)
        # Multiple passes through the graph to account for shifts in balanced branches
        while imbalanced > 0:
            insert_proxy_terms(dag, dag[go_root], original_size)
            imbalanced = sum(is_imbalanced(term) for term in dag.values())
            print(f"Proxy terms added: {len(dag) - dag_size}")
            print(f"Imbalanced terms left: {imbalanced}")
            dag_size = len(dag)

        print(f"Total amount of inserted balancing proxies: {len(dag) - original_size}")

    def test_plot_merge_results(self):
        go_ref = copy_dag(go_bp_main)
        leaf_ids = all_leaf_ids(go_ref)

        # Depth distribution for different merge conditions
        plot_depth_distribution(go_ref, leaf_ids, title="Distribution of GO-leaf depths\nfor varying merge conditions")
        plt.legend(["Original"], title="(#parents, #children)")
        plt.show()

        merge_conditions = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        for merge_condition in merge_conditions:
            go = copy_dag(go_bp_main)
            pruning_events, merge_events = 1, 1
            while pruning_events + merge_events > 0:
                merge_events = merge_chains(go, merge_condition[0], merge_condition[1])
                pruning_events = prune_skip_connections(go)
            plot_depth_distribution(go_ref, leaf_ids)
            plot_depth_distribution(go, leaf_ids, title="Distribution of GO-leaf depths\nfor varying merge conditions")
            plt.legend(["Original", merge_condition], title="(#parents, #children)")
            plt.show()

    def test_plot_merge_proxy_results(self):
        go_ref = copy_dag(go_bp_main)
        leaf_ids = all_leaf_ids(go_ref)
        # Depth distribution for different merge conditions, with balancing
        plot_depth_distribution(go_ref, leaf_ids,
                                title="Distribution of GO-leaf depths\nfor varying merge conditions (balanced)")
        plt.legend(["Original"], title="(#parents, #children)")
        plt.show()

        merge_conditions = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        for merge_condition in merge_conditions:
            go = copy_dag(go_bp_main)
            pruning_events, merge_events = 1, 1
            while pruning_events + merge_events > 0:
                merge_events = merge_chains(go, merge_condition[0], merge_condition[1])
                pruning_events = prune_skip_connections(go)

            original_size = len(go)
            go_root = "GO:0008150"
            imbalanced = sum(is_imbalanced(term) for term in go.values())
            dag_size = original_size
            while imbalanced > 0:
                insert_proxy_terms(go, go[go_root], original_size)
                imbalanced = sum(is_imbalanced(term) for term in go.values())

                # when imbalanced, yet no more proxies are added, stop (to stop infinite loop)
                if len(go) - dag_size == 0:
                    print("ERROR: Imbalance cannot be resolved, infinite loop terminated!")
                    break

                print(f"Proxy terms added: {len(go) - dag_size}")
                print(f"Imbalanced terms left: {imbalanced}")
                dag_size = len(go)

            plot_depth_distribution(go_ref, leaf_ids)
            plot_depth_distribution(go, leaf_ids,
                                    title="Distribution of GO-leaf depths\nfor varying merge conditions (balanced)")
            plt.legend(["Original", merge_condition], title="(#parents, #children)")
            plt.show()

    def test_pull_leaves_down_small(self):
        dag = DAGGenerator.dag3()
        pull_leaves_down(dag, len(dag))
        self.assertEqual(dag["C"].depth, dag["E"].depth)

    def test_pull_leaves_down_go(self):
        go = copy_dag(go_bp_main)
        pull_leaves_down(go, len(go))
        print_dag_info(go)
        max_depth = max(go[leaf_id].depth for leaf_id in all_leaf_ids(go))
        depth_sum = sum(go[leaf_id].depth for leaf_id in all_leaf_ids(go))
        self.assertEqual(depth_sum, max_depth * len(all_leaf_ids(go)))

    def test_insert_proxy_pull_leaves(self):
        dag = copy_dag(go_bp_main)
        original_size = len(dag)
        go_root = "GO:0008150"
        imbalanced = sum(is_imbalanced(term) for term in dag.values())
        dag_size = original_size
        leaf_ids = all_leaf_ids(dag)
        plot_depth_distribution(dag, leaf_ids)
        # Balancing iterations
        while imbalanced > 0:
            insert_proxy_terms(dag, dag[go_root], original_size)
            imbalanced = sum(is_imbalanced(term) for term in dag.values())
            print(f"Proxy terms added: {len(dag) - dag_size}")
            print(f"Imbalanced terms left: {imbalanced}")
            dag_size = len(dag)
        print("COMPLETED: Balancing DAG with proxies")

        print(f"\nNumber of original non-leaf terms: {original_size - len(leaf_ids)}")
        print(f"Number of leaves: {len(all_leaf_ids(dag))}")
        print(f"Number of inserted balancing proxies: {len(dag) - original_size}")
        average_leaf_depth = sum(dag[leaf_id].depth for leaf_id in all_leaf_ids(dag)) / len(all_leaf_ids(dag))
        print(f"Average leaf depth: {average_leaf_depth:.1f}\n")

        pull_leaves_down(dag, original_size)

    def test_merge_prune_pull_leaves(self):
        go = copy_dag(go_bp_main)
        # Plot leaf depth distribution before and after merge-prune convergence
        leaf_ids = all_leaf_ids(go)
        plot_depth_distribution(go, leaf_ids)
        # First merge, then prune:
        pruning_events = 1
        merge_events = 1
        parent_threshold = 1
        children_threshold = 1
        while pruning_events + merge_events > 0:
            merge_events = merge_chains(go, parent_threshold, children_threshold)
            pruning_events = prune_skip_connections(go)
        print("COMPLETED: Merge-prune until convergence")
        pull_leaves_down(go, len(go))

    def test_load_relationships(self):
        go_rel_full = GODag("go-basic.obo", optional_attrs={"relationship"})
        go_rel = filter_go_by_namespace(go_rel_full, "biological_process")
        del go_rel_full
        self.assertIsNotNone(go_rel["GO:0002893"].relationship)
        pass

    def test_leaf_relationships(self):
        go = copy_dag(go_bp_main)
        go_rel_full = copy_dag(go_rel_main)
        go_rel = filter_go_by_namespace(go_rel_full, "biological_process")
        del go_rel_full
        leaf_ids = all_leaf_ids(go)
        leaf_ids_with_rel = [leaf_id for leaf_id in leaf_ids if len(go_rel[leaf_id].relationship_rev) > 0]
        print(f"{len(leaf_ids_with_rel)}/{len(leaf_ids)} leaves have children through relationships.")

    def test_relationships_to_parents(self):
        go_bp_rel = filter_go_by_namespace(copy_dag(go_rel_main), "biological_process")
        relationships_to_parents(go_bp_rel)
        self.assertEqual(len(go_bp_rel["GO:0000027"].children), 3)
        self.assertEqual(len(go_bp_rel["GO:0000027"].parents), 3)

    def test_plot_relationship_effect(self):
        go_ref = copy_dag(go_main)
        go_rel_ref = copy_dag(go_rel_main)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        plot_depth_distribution(go_ref, go_ref.keys(), sub_fig=ax1,
                                title="Only 'is_a' relationship")
        ax1.legend(["Original"], title="(#p, #c)")
        plot_depth_distribution(go_rel_ref, go_rel_ref.keys(), sub_fig=ax2,
                                title="All relationships")
        ax2.legend(["Original"], title="(#p, #c)")
        ax2.set_ylabel("")
        fig.suptitle("Distribution of GO-term depths for varying merge conditions")
        plt.show()

        merge_conditions = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
        for merge_condition in merge_conditions:
            go = copy_dag(go_ref)
            go_rel = copy_dag(go_rel_ref)
            relationships_to_parents(go_rel)
            merge_prune_until_convergence(go, merge_condition[0], merge_condition[1])
            merge_prune_until_convergence(go_rel, merge_condition[0], merge_condition[1])
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_depth_distribution(go_ref, go_ref.keys(), sub_fig=ax1)
            plot_depth_distribution(go, go.keys(), sub_fig=ax1,
                                    title="Only 'is_a' relationship")
            plot_depth_distribution(go_rel_ref, go_rel_ref.keys(), sub_fig=ax2)
            plot_depth_distribution(go_rel, go_rel.keys(), sub_fig=ax2,
                                    title="All relationships")
            ax1.legend(["Original", merge_condition], title="(#p, #c)")
            ax2.legend(["Original", merge_condition], title="(#p, #c)")
            ax2.set_ylabel("")
            fig.suptitle("Distribution of GO-term depths for varying merge conditions")
            plt.show()

    def test_merge_prune_balance_pull_bp(self):
        go = copy_dag(go_bp_main)
        merge_prune_until_convergence(go, 1, 1)
        balance_until_convergence(go)
        pull_leaves_down(go, len(go_bp_main))
        print_dag_info(go)
