from goatools.obo_parser import GOTerm


class ProxyTerm(GOTerm):

    def __init__(self, item_id: str, parents: set[GOTerm], children: set[GOTerm]):
        super().__init__()
        self.item_id = item_id
        self.parents = parents
        self.children = children
        self.level = 0
        self.depth = 0
        self.namespace = "biological_process"

        for parent in self.parents:
            parent.children.add(self)
            parent.children -= self.children
        for child in self.children:
            child.parents.add(self)
            child.parents -= self.parents
