from utils import Dataset, PointFactory

import matplotlib.pyplot as plt
import numpy as np


class DummyRandomPoints(PointFactory):

    def __init__(self,
                 points_per_blob: int = 100,
                 loc: float = 0.0,
                 scale: float = 0.6,
                 ):
        self._points_per_blob = points_per_blob
        self._loc = loc
        self._scale = scale

    def get_points(self):
        result = np.random.normal(loc=self._loc, scale=self._scale, size=self._points_per_blob)
        result.sort()
        return result


class DummyRandomLeftPoints(DummyRandomPoints):

    def __init__(self,
                 points_per_blob: int = 100,
                 scale: float = 0.6,
                 ):
        super().__init__(loc=-3.0, scale=scale, points_per_blob=points_per_blob)


class DummyRandomCenterPoints(DummyRandomPoints):

    def __init__(self,
                 points_per_blob: int = 100,
                 scale: float = 0.6,
                 ):
        super().__init__(loc=0.0, scale=scale, points_per_blob=points_per_blob)


class DummyRandomRightPoints(DummyRandomPoints):

    def __init__(self,
                 points_per_blob: int = 100,
                 scale: float = 0.6,
                 ):
        super().__init__(loc=3.0, scale=scale, points_per_blob=points_per_blob)


class DummyDatasetForRegression(Dataset):

    @classmethod
    def get_point_factories(cls, points_per_blob: int = 100, scale: float = 0.6):
        return (
            DummyRandomLeftPoints(points_per_blob=points_per_blob, scale=scale),
            DummyRandomCenterPoints(points_per_blob=points_per_blob, scale=scale),
            DummyRandomRightPoints(points_per_blob=points_per_blob, scale=scale),
        )

    @classmethod
    def get_functions(cls):
        return (
            lambda x: 2 * x + 3,
            lambda x: -2 * x - 1,
            lambda x: x ** 2 - 6 * x + 5,
        )

    def __init__(self, points_per_blob: int = 100, scale: float = 0.6):
        super().__init__(
            point_factories=self.get_point_factories(points_per_blob=points_per_blob, scale=scale),
            functions=self.get_functions(),
        )

    def visualize_graphs(self, graph_alpha=0.1):
        left_dataset, center_dataset, right_dataset = self.dataset_collection
        plt.title('Function graphs')
        plt.ylabel('y')
        plt.scatter(left_dataset.x, left_dataset.y, alpha=graph_alpha)
        plt.scatter(center_dataset.x, center_dataset.y, alpha=graph_alpha)
        plt.scatter(right_dataset.x, right_dataset.y, alpha=graph_alpha)

    def visualize_histograms(self, hist_alpha=0.5, bins=50):
        left_dataset, center_dataset, right_dataset = self.dataset_collection
        plt.title('Values histograms')
        plt.xlabel('x')
        plt.ylabel('Density value')
        plt.hist(left_dataset.x, alpha=hist_alpha, density=True, bins=bins)
        plt.hist(center_dataset.x, alpha=hist_alpha, density=True, bins=bins)
        plt.hist(right_dataset.x, alpha=hist_alpha, density=True, bins=bins)

    def visualize_dataset(self, graph_alpha=0.1, hist_alpha=0.5, bins=50):
        plt.figure(figsize=(7, 6))
        plt.subplot(2, 1, 1)
        self.visualize_graphs(graph_alpha=graph_alpha)
        plt.subplot(2, 1, 2)
        self.visualize_histograms(hist_alpha=hist_alpha, bins=bins)

    def visualize_model(self, model, start=-7, stop=7, steps=1000, graph_alpha=0.1, prediction_alpha=0.5):
        self.visualize_graphs(graph_alpha=graph_alpha)
        full_x = np.linspace(start=start, stop=stop, num=steps)
        predictions = model.predict(full_x, batch_size=steps)
        plt.plot(full_x, predictions, alpha=prediction_alpha, c='r', label='Prediction')
        plt.legend()

    def visualize_variational_model(
            self,
            model,
            nb_of_samples,
            start=-7,
            stop=7,
            steps=1000,
            graph_alpha=0.1,
            prediction_alpha=0.5,
    ):
        full_x = np.linspace(start=start, stop=stop, num=steps)
        samples = np.hstack([model.predict(full_x, batch_size=steps) for _ in range(nb_of_samples)])
        means = samples.mean(axis=1)
        stds = samples.std(axis=1)
        low_confidence_itervals = means - stds
        high_confidence_itervals = means + stds
        plt.plot(full_x, means, alpha=prediction_alpha, c='r', label='Prediction')
        plt.plot(full_x, low_confidence_itervals, alpha=prediction_alpha, c='c', label='Low confidence interval')
        plt.plot(full_x, high_confidence_itervals, alpha=prediction_alpha, c='c', label='High confidence interval')
        plt.fill_between(full_x, low_confidence_itervals, high_confidence_itervals, color='c')
        self.visualize_graphs(graph_alpha=graph_alpha)
        plt.legend()

    def visualize_prior_ensemble_model(
            self,
            model,
            start=-7,
            stop=7,
            steps=1000,
            graph_alpha=0.1,
            prediction_alpha=0.5,
    ):
        full_x = np.linspace(start=start, stop=stop, num=steps)
        samples = model.predict(full_x)
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)
        low_confidence_itervals = means - stds
        high_confidence_itervals = means + stds
        plt.plot(full_x, means, alpha=prediction_alpha, c='r', label='Prediction')
        plt.plot(full_x, low_confidence_itervals, alpha=prediction_alpha, c='c', label='Low confidence interval')
        plt.plot(full_x, high_confidence_itervals, alpha=prediction_alpha, c='c', label='High confidence interval')
        plt.fill_between(full_x, low_confidence_itervals, high_confidence_itervals, color='c')
        self.visualize_graphs(graph_alpha=graph_alpha)
        plt.legend()
