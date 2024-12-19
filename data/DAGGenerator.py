from goatools.obo_parser import GOTerm


class DAGGenerator:
    """Class for small DAG examples."""

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    @staticmethod
    def dag1():
        a = GOTerm("biological_process")
        b = GOTerm("biological_process")
        c = GOTerm("biological_process")
        d = GOTerm("biological_process")
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        d.item_id = "D"
        dag = {"A": a, "B": b, "C": c, "D": d}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c, d}
        c.parents = {a, b}
        d.parents = {b}
        a.depth, b.depth, c.depth, d.depth = 0, 1, 2, 2
        a.level, b.level, c.level, d.level = 0, 1, 1, 2

        return dag

    #      A
    #    / |
    #  /   B
    # | /
    # C
    @staticmethod
    def dag2():
        a = GOTerm("biological_process")
        b = GOTerm("biological_process")
        c = GOTerm("biological_process")
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        dag = {"A": a, "B": b, "C": c}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c}
        c.parents = {a, b}
        a.depth, b.depth, c.depth = 0, 1, 2
        a.level, b.level, c.level = 0, 1, 1

        return dag

    #       A
    #     / |
    #   /   B
    #  | /    \
    #  C       D
    #           \
    #            E
    @staticmethod
    def dag3():
        a, b, c, d, e = (GOTerm("biological_process") for _ in range(5))
        a.item_id = "A"
        b.item_id = "B"
        c.item_id = "C"
        d.item_id = "D"
        e.item_id = "E"
        dag = {"A": a, "B": b, "C": c, "D": d, "E": e}

        a.children = {b, c}
        b.parents = {a}
        b.children = {c, d}
        c.parents = {a, b}
        d.parents = {b}
        d.children = {e}
        e.parents = {d}
        a.depth, b.depth, c.depth, d.depth, e.depth = 0, 1, 2, 2, 3
        a.level, b.level, c.level, d.level, e.level = 0, 1, 1, 2, 3

        return dag
