from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessing components.

    This class defines the standard interface for all data preprocessors, following
    the scikit-learn transformer pattern. It ensures that all concrete preprocessor
    implementations provide consistent methods for fitting to training data,
    transforming data, and performing both operations in sequence.

    The class enforces a two-phase preprocessing workflow:
    1. Fit phase: Learn parameters from training data
    2. Transform phase: Apply learned transformations to new data

    This pattern enables proper handling of train/validation/test splits by ensuring
    that preprocessing parameters are learned only from training data and then
    consistently applied to all datasets.

    Abstract Methods:
        fit(df): Learn preprocessing parameters from training data
        transform(df): Apply learned transformations to data
        fit_transform(df): Convenience method combining fit and transform

    Example:
        >>> class StandardScaler(BasePreprocessor):
        ...     def __init__(self):
        ...         self.mean_ = None
        ...         self.std_ = None
        ...
        ...     def fit(self, df):
        ...         self.mean_ = df.mean()
        ...         self.std_ = df.std()
        ...         return self
        ...
        ...     def transform(self, df):
        ...         return (df - self.mean_) / self.std_
        ...
        ...     def fit_transform(self, df):
        ...         return self.fit(df).transform(df)
        >>>
        >>> scaler = StandardScaler()
        >>> train_scaled = scaler.fit_transform(train_df)
        >>> test_scaled = scaler.transform(test_df)  # Uses train statistics

    Note:
        - Requires 'from abc import ABC, abstractmethod' to be imported
        - Concrete implementations must override all abstract methods
        - Following scikit-learn conventions, fit() should return self for chaining
        - Transform methods should not modify the original data
        - fit_transform() is typically implemented as self.fit(df).transform(df)
    """

    @abstractmethod
    def fit(self, df):
        """
        Learn preprocessing parameters from the input data.

        This method analyzes the training data to compute and store any parameters
        needed for transformation (e.g., means, standard deviations, encodings).
        It should not modify the input data, only extract necessary statistics.

        Args:
            df: Input data to learn from. Type depends on the specific preprocessor
                implementation (pandas DataFrame, numpy array, etc.).

        Returns:
            self: Returns the fitted preprocessor instance to enable method chaining.

        Note:
            - Should only be called on training data to avoid data leakage
            - May be called multiple times to refit on new training data
            - Should store learned parameters as instance attributes
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Apply the learned preprocessing transformation to data.

        This method applies the preprocessing transformation using parameters
        learned during the fit phase. It can be called on training, validation,
        or test data using the same learned parameters.

        Args:
            df: Input data to transform. Should have the same structure as the
                data used during fitting.

        Returns:
            Transformed data in the same format as the input. The exact return
            type depends on the specific preprocessor implementation.

        Raises:
            NotFittedError: If transform is called before fit (implementation-dependent).

        Note:
            - Must be called after fit() to ensure parameters are available
            - Should not modify the original input data
            - Should handle edge cases like unseen categories gracefully
        """
        pass

    @abstractmethod
    def fit_transform(self, df):
        """
        Fit the preprocessor and transform the data in one step.

        This convenience method combines the fit and transform operations,
        learning parameters from the input data and immediately applying
        the transformation. Equivalent to calling fit(df).transform(df).

        Args:
            df: Input data to fit on and transform. Typically training data.

        Returns:
            Transformed data in the same format as the input.

        Note:
            - Primarily used on training data during the initial preprocessing
            - More efficient than separate fit() and transform() calls
            - Should produce the same result as fit(df).transform(df)
        """
        pass
