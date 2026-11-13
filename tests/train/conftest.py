"""Fixtures for the training tests."""

import socket

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def worker_port() -> int:
    """A free TCP port for a torchrun rendezvous.

    Each distributed test gets its own so consecutive runs cannot collide on a
    port still in TIME_WAIT.
    """
    return _free_port()
