"""Application-layer least-autonomy enforcement.

Only the Lookup Agent is authorized to invoke check_order_status.
"""

class GovernanceAuthorizationError(PermissionError):
    """Raised when an agent attempts an unauthorized tool invocation."""


LOOKUP_AGENT = "Lookup Agent"
LOOKUP_TOOL = "check_order_status"

AUTHORIZED_TOOLS = {
    LOOKUP_AGENT: {LOOKUP_TOOL},
    "Retrieval Agent": set(),
    "Response Composer": set(),
}


def authorize_tool_call(agent_role: str, tool_name: str) -> bool:
    """Allow only explicitly authorized agent/tool pairs."""
    allowed = tool_name in AUTHORIZED_TOOLS.get(agent_role, set())
    if not allowed:
        raise GovernanceAuthorizationError(
            f"Least-autonomy policy blocked {agent_role!r} from calling {tool_name!r}. "
            f"Only {LOOKUP_AGENT} may call {LOOKUP_TOOL}."
        )
    return True


def can_call_tool(agent_role: str, tool_name: str) -> bool:
    """Non-throwing authorization check useful for wiring/tests."""
    return tool_name in AUTHORIZED_TOOLS.get(agent_role, set())
