from goatools.obo_parser import GOTerm


class DummyTerm(GOTerm):

    def __init__(self, item_id: str, parents: set[GOTerm], children: set[GOTerm], depth: int):
        super().__init__()
        self.item_id = item_id
        self.parents = parents
        self.children = children
