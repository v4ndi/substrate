"""The sampler contract: wrap the record stream and drop what fails."""


class BaseSampler:
    """The sampler contract: decide which records survive.

    A sampler runs during the *scan*, before the payload is read, so filtering
    saves IO and not just compute. Its decisions define how many valid records
    each file holds, which is what sharding is planned from — so a sampler must
    be deterministic.
    """

    def set_dataset_iterator(self, dataset_iterator):
        self.dataset_iterator = dataset_iterator

    def __iter__(self):
        raise NotImplementedError("Method must be implemented by child classes")
