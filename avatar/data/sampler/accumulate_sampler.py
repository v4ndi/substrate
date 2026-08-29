class AccumulateSampler:
    def __init__(self, batch_size: int, accumulated_batches: int = 2):
        self.batch_size = batch_size
        self.accumulated_batches = accumulated_batches

    def set_dataset_iterator(self, dataset_iterator):
        self.dataset_iterator = dataset_iterator

    def _fill_buffers(self):
        self.buff = []
        self.continue_iter_flag = False
        for data in self.dataset_iterator:
            self.buff.append(data)
            if len(self.buff) >= self.batch_size * self.accumulated_batches:
                self.continue_iter_flag = True
                break

    def __iter__(self):
        self.continue_iter_flag = True
        assert self.dataset_iterator is not None, (
            "set dataset iterator, using set_dataset_iterator() method"
        )

        while self.continue_iter_flag:
            self._fill_buffers()
            yield from self.buff
