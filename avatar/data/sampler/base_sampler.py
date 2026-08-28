class BaseSampler:
    def set_dataset_iterator(self, dataset_iterator):
        self.dataset_iterator = dataset_iterator

    def __iter__(self):
        raise NotImplementedError("Method must be implemented by child classes")
