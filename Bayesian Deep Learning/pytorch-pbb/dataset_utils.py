import torch
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes


class DatasetCache(DatasetWrapper):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.cache = {}

    def __getitem__(self, index):
        with torch.no_grad():
            if index in self.cache:
                return self.cache[index]
            else:
                self.cache[index] = self.dataset[index]
                return self.cache[index]


class DatasetSubset(DatasetWrapper):
    def __init__(self, dataset, start=0, stop=None):
        super().__init__(dataset)
        self.start = start
        self.stop = stop
        if stop is not None:
            self.stop = min(self.stop, len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[index + self.start]

    def __len__(self):
        stop = self.stop or len(self.dataset)
        return stop - self.start


class DatasetWhiten(DatasetWrapper):
    def __init__(self, dataset):
        super().__init__(dataset)

        all_unwhite_x = torch.stack([self.dataset[i][0]
                                     for i in range(len(self.dataset))])
        self.mean, self.std = all_unwhite_x.mean(), all_unwhite_x.std()

    def __getitem__(self, index):
        return (self.dataset[index] - self.mean) / self.std
