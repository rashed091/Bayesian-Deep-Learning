from typing import Callable, Iterable

import numpy as np
from matplotlib import pyplot as plt

from cached_property import cached_property


class PointFactory:

    def get_points(self) -> np.array:
        raise NotImplementedError


class XYPair:

    def __init__(self, x: np.array, y: np.array):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @classmethod
    def merge(cls, collection: Iterable['XYPair']):
        return cls(
            x=np.hstack([xy_pair.x for xy_pair in collection]),
            y=np.hstack([xy_pair.y for xy_pair in collection]),
        )


class Dataset:

    def __init__(
            self,
            point_factories: Iterable[PointFactory],
            functions: Iterable[Callable],
    ):
        self._point_factories = point_factories
        self._functions = functions

    @cached_property
    def points_collection(self):
        return [point_factory.get_points() for point_factory in self._point_factories]

    @cached_property
    def all_points(self):
        return np.vstack(self.points_collection)

    @cached_property
    def dataset_collection(self):
        points_collections = self.points_collection
        return [XYPair(x=points, y=function(points))
                for points, function in zip(points_collections, self._functions)]

    @cached_property
    def full_dataset(self):
        return XYPair.merge(self.dataset_collection)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_reverse(x):
    return np.log(x) - np.log(1 - x)


def concrete_sigmoid(p, u, t):
    return sigmoid(1 / t * (sigmoid_reverse(u) + sigmoid_reverse(p)))


def plot_concrete_sigmoid(p, t, step):
    x = np.linspace(1 / step, 1 - 1 / step, step - 2)
    plt.xlabel('p')
    plt.ylabel('Value')
    values = concrete_sigmoid(p, x, t)
    plt.title(f'Mean = {values.mean()}')
    plt.plot(x, values)

