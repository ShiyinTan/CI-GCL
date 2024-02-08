from GCL.augmentors.augmentor import Graph, Augmentor


class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph, **kwargs) -> Graph:
        return g
