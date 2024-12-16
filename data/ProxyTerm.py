from goatools.obo_parser import GOTerm


class ProxyTerm(GOTerm):

    def __init__(self, item_id: str, parents: set[GOTerm], children: set[GOTerm]):
        super().__init__()
        self.item_id = item_id
        self.parents = parents
        self.children = children
        self.level = 0
        self.depth = 0

        for parent in self.parents:
            parent.children.add(self)
            parent.children -= self.children
        for child in self.children:
            child.parents.add(self)
            child.parents -= self.parents

        self._set_level_and_depth()

    def _set_level_and_depth(self):
        """For any (proxy)term inserted into the DAG, set its level and depth, and update children"""
        parents = self.parents
        self.level = min(parent.level for parent in parents) + 1
        self.depth = max(parent.depth for parent in parents) + 1

        for child in self._get_all_children():
            child.level = child.level + 1
            child.depth = child.depth + 1

    def _get_all_children(self):
            all_children = set()
            for child in self.children:
                all_children.add(child)
                all_children |= child.get_all_children()
            return all_children
