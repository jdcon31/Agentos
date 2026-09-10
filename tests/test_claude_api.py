"""Tests for the frontier client's retry policy. No network, no API key."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tools import claude_api as api


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_transient_statuses_are_retryable(status):
    assert api.is_retryable(_StatusError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    assert api.is_retryable(_StatusError(status)) is False


@pytest.mark.parametrize("message", [
    "Error: overloaded_error",
    "rate limit exceeded",
    "upstream UNAVAILABLE",
    "request timeout after 60s",
])
def test_retryable_by_message_when_no_status(message):
    assert api.is_retryable(Exception(message)) is True


def test_unrelated_error_is_not_retryable():
    assert api.is_retryable(ValueError("bad prompt")) is False


def test_backoff_doubles_each_attempt():
    delays = [api.backoff_delay(n, base=2.0) for n in range(4)]
    assert delays == [2.0, 4.0, 8.0, 16.0]


def test_backoff_respects_a_custom_base():
    assert api.backoff_delay(0, base=0.5) == 0.5
    assert api.backoff_delay(3, base=0.5) == 4.0
