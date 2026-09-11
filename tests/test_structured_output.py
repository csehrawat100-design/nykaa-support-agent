"""Task 9 structured-output tests."""

import pytest
from pydantic import ValidationError

from agents.schemas import ResponseFormat
from agents.crew import validate_crew_response


class FakeResult:
    def __init__(self, raw):
        self.raw = raw
        self.pydantic = None
        self.json_dict = None


def test_response_format_accepts_valid_payload():
    result = validate_crew_response(
        FakeResult(
            '{"response":"Order is shipped.","source_type":"order","grounded":true}'
        )
    )
    assert isinstance(result, ResponseFormat)
    assert result.source_type == "order"
    assert result.grounded is True


def test_response_format_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        validate_crew_response(
            FakeResult('{"response":"Order is shipped.","grounded":true}')
        )


def test_response_format_rejects_unknown_field():
    with pytest.raises(ValidationError):
        validate_crew_response(
            FakeResult(
                '{"response":"Order is shipped.","source_type":"order",'
                '"grounded":true,"extra":"not allowed"}'
            )
        )


def test_response_format_rejects_invalid_source_type():
    with pytest.raises(ValidationError):
        validate_crew_response(
            FakeResult(
                '{"response":"Order is shipped.","source_type":"database",'
                '"grounded":true}'
            )
        )
