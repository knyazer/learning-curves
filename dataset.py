import lcdb
import torch
from typing import Callable, List, Any, Dict
from tqdm import tqdm


"""
LCDB provides the worst every api to access the data,
so we are just going to write a simple-ish wrapper around it,
that allows for a convenient filtering.

The idea is to allow for things like ```
dataset = LCDB()

dataset \\
    .filter(lambda x: x.dataset == 'blahblah') \\
    .filter(lambda x: x.model == 'linearregression')

since this is much more convenient
"""

class Curve:
    """
    A single instance of a learning curve;
    contains 3 sets of the outputs indexed by anchors:
    - train, val and test

    in addition, we, of course, include the list of anchors:
    - anchors

    and the dataset and model that generated this curve:
    - dataset: str, model: str

    So the issue here is that we are losing the ability to
    distinguish between different outer/inner seeds, etc
    but I don't care currently: let's just consider all of them
    to be simply different augmentations of the original one.

    However, for reasons of better in-domain testing, I include
    'seed' that is representative of the split (combination of
    outer and inner seeds) that was used to generate this curve.
    """
    anchors: torch.Tensor
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor

    dataset: int
    learner: int
    seed: int

    def __init__(self, anchors, train, val, test, dataset, learner, seed) -> None:
        self.anchors = anchors
        self.train = train
        self.val = val
        self.test = test
        self.dataset = dataset
        self.learner = learner
        self.seed = seed


class LCDB:
    """
    Does the processing of the curves, and overall is the main entry point.

    Allows to get meta-features of datasets, curves, etc, etc...
    """

    curves: List[Curve]
    datasets: List[int]
    learners: List[str]
    meta_features: Dict[str, Dict[str, Any]]

    def load(self):
        self.curves = []
        all_curves = lcdb.get_all_curves()
        self.learners = (all_curves["learner"].unique())
        self.datasets = (all_curves["openmlid"].unique())
        self.datasets = [int(x) for x in self.datasets]






    def filter(self, f: Callable[[Curve], bool]):
        return LCDB([curve for curve in self.curves if f(curve)])


if __name__ == "__main__":
    dataset = LCDB()
    dataset.load()
    print(dataset.curves)


