import random
import numpy as np
from sklearn import preprocessing
X, Y = 0, 1
TOTAL_SIZE = 70_000
FIXED_SIZE = 10_000
EVALUATION_SIZE = 1_000
SHADOW_SIZE = 59_000

def get_target(dataset, seed: int = 123456):
    random.seed(seed)
    all = dataset[X]
    labels = dataset[Y]
    before_size = len(all) # assuming both samples and labels are same lenght

    target_int = random.randint(0, len(all))
    target = (np.delete(all, target_int, 0), np.delete(labels, target_int, 0))
    assert (len(all) == before_size - 1) and (len(labels) == before_size)
    return (target, all)


def get_shadow(dataset, seed: int = 123456):
    """
    Get the shadow target randomly, delete it from the dataset
    :param dataset: The dataset which will be remove from
    :param seed: Seed value for random reproducibility
    :return: Tuple value representing the shadow and the rest of the dataset
    """
    sample = dataset[0]
    label = dataset[1]

    random.seed(seed)
    shadow_int = random.randint(0, len(sample))
    shadow = (np.delete(sample, shadow_int), np.delete(label, shadow_int))
    return shadow, dataset

def rescale(dataset):
