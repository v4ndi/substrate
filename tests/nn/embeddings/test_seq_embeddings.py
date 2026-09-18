import pytest
import torch
import torch.nn as nn

from fmlib.nn.embedding import BaseEventSequenceEmbedding, EventSequenceEmbedding


class TestBaseEventSequenceEmbedding:
    """Tests for BaseEventSequenceEmbedding class"""

    @pytest.fixture
    def valid_columns_meta(self):
        return {
            "mcc": {"type": "categorical", "n_classes": 350},
            "price": {"type": "numeric", "n_classes": 1},
        }

    @pytest.fixture
    def invalid_columns_meta(self):
        return {
            "mcc": {"type": "invalid_type", "n_classes": 350},  # Invalid type
            "price": {"type": "numeric", "n_classes": 0},  # Invalid n_classes
        }

    def test_init_with_valid_columns_meta(self, valid_columns_meta):
        """Test initialization with valid columns_meta"""
        embedding = BaseEventSequenceEmbedding(
            hidden_size=64, columns_meta=valid_columns_meta
        )
        assert embedding.hidden_size == 64
        assert embedding.columns_meta == valid_columns_meta

    def test_init_with_invalid_columns_meta(self, invalid_columns_meta):
        """Test initialization raises assertion for invalid columns_meta"""
        with pytest.raises(AssertionError):
            BaseEventSequenceEmbedding(
                hidden_size=64, columns_meta=invalid_columns_meta
            )

    def test_columns_property(self, valid_columns_meta):
        """Test columns property returns correct keys"""
        embedding = BaseEventSequenceEmbedding(
            hidden_size=64, columns_meta=valid_columns_meta
        )
        assert set(embedding.columns) == set(valid_columns_meta.keys())

    def test_check_columns_meta(self, valid_columns_meta, invalid_columns_meta):
        """Test _check_columns_meta static method"""
        assert BaseEventSequenceEmbedding._check_columns_meta(valid_columns_meta)
        assert not BaseEventSequenceEmbedding._check_columns_meta(invalid_columns_meta)

        # Test missing required keys
        missing_keys_meta = {"mcc": {"type": "categorical"}}  # Missing n_classes
        assert not BaseEventSequenceEmbedding._check_columns_meta(missing_keys_meta)


class TestEventSequenceEmbedding:
    """Tests for EventSequenceEmbedding class"""

    @pytest.fixture
    def valid_columns_meta(self):
        return {
            "mcc": {"type": "categorical", "n_classes": 350},
            "price": {"type": "numeric", "n_classes": 1},
        }

    @pytest.fixture
    def sample_features(self):
        return {
            "mcc": torch.randint(0, 350, (1, 10)),  # Batch of 10 categorical values
            "price": torch.randn(1, 10),  # Batch of 10 numeric values
        }

    def test_init_creates_embedding_layers(self, valid_columns_meta):
        """Test initialization creates correct embedding layers"""
        embedding = EventSequenceEmbedding(
            hidden_size=64, columns_meta=valid_columns_meta
        )

        assert isinstance(embedding.embedding_layer, nn.ModuleDict)
        assert "mcc" in embedding.embedding_layer
        assert "price" in embedding.embedding_layer
        assert isinstance(embedding.embedding_layer["mcc"], nn.Embedding)
        assert hasattr(
            embedding.embedding_layer["price"], "weight"
        )  # Checking LinearEmbeddings

    def test_forward_pass(self, valid_columns_meta, sample_features):
        """Test forward pass returns correct shape"""
        hidden_size = 64
        embedding = EventSequenceEmbedding(
            hidden_size=hidden_size, columns_meta=valid_columns_meta
        )

        output = embedding(sample_features)

        # Check output shape: [batch_size, hidden_size, num_columns]
        assert output.shape == (1, 10, 2, hidden_size)

    def test_forward_with_missing_column(self, valid_columns_meta):
        """Test forward raises error with missing feature column"""
        embedding = EventSequenceEmbedding(
            hidden_size=64, columns_meta=valid_columns_meta
        )

        with pytest.raises(KeyError):
            embedding({"mcc": torch.randint(0, 350, (10,))})  # Missing price
