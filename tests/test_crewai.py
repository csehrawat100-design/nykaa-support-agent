"""Task 7 structural tests.

These tests require the project's installed CrewAI dependency and the completed
Tasks 3–6 modules.
"""

from agents.crew import (
    CALIBRATED_THRESHOLD,
    ResponseFormat,
    RAGLookupTool,
    OrderLookupTool,
    build_crew,
)


def test_selected_configuration():
    assert CALIBRATED_THRESHOLD == 0.4844
    assert RAGLookupTool().args_schema.model_fields["query"]


def test_lookup_tool_schema():
    assert OrderLookupTool().args_schema.model_fields["record_id"]


def test_composer_has_no_tools():
    crew = build_crew()
    composer = next(a for a in crew.agents if a.role == "Response Composer")
    assert composer.tools == []


def test_response_schema():
    value = ResponseFormat(
        response="supported",
        source_type="policy",
        grounded=True,
    )

    assert value.response == "supported"
    assert value.source_type == "policy"
    assert value.grounded is True