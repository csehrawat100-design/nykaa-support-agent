"""Deterministic MockLLM for the Nykaa Support Agent CrewAI layer."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from crewai import BaseLLM


ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b")

FALLBACK_MESSAGE = (
    "I don't know based on the available Nykaa support knowledge base."
)


class MockLLM(BaseLLM):
    """Local deterministic LLM used for the CrewAI integration."""

    def __init__(
        self,
        model: str = "mock-llm",
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
        )

        self.tool_results: list[dict[str, Any]] = []

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    # ------------------------------------------------------------------
    # Main CrewAI entry point
    # ------------------------------------------------------------------

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> str:

        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages,
                }
            ]

        task_text = self._get_task_text(messages)
        agent_role = self._get_agent_role(kwargs)

        print("\n========== MOCK_LLM DEBUG ==========")
        print("AGENT ROLE:")
        print(agent_role)

        print("\nTASK TEXT:")
        print(task_text)

        print("\nAVAILABLE FUNCTIONS:")
        print(available_functions)

        print("\nTOOLS:")
        print(tools)

        print("====================================\n")

        # --------------------------------------------------------------
        # Retrieval Agent
        # --------------------------------------------------------------

        if self._is_retrieval_agent(agent_role):

            result = self._run_rag(task_text)

            self.tool_results.append(
                {
                    "type": "rag",
                    "payload": result,
                }
            )

            return self._final_answer(result)

        # --------------------------------------------------------------
        # Lookup Agent
        # --------------------------------------------------------------

        if self._is_lookup_agent(agent_role):

            order_match = ORDER_ID_RE.search(task_text)

            if not order_match:
                result = {
                    "response": (
                        "No order lookup is needed because "
                        "the request does not contain an order ID."
                    ),
                    "source_type": "fallback",
                    "grounded": False,
                }

                self.tool_results.append(
                    {
                        "type": "lookup",
                        "payload": result,
                    }
                )

                return self._final_answer(result)

            result = self._run_order_lookup(
                order_match.group(0)
            )

            self.tool_results.append(
                {
                    "type": "lookup",
                    "payload": result,
                }
            )

            return self._final_answer(result)

        # --------------------------------------------------------------
        # Response Composer
        # --------------------------------------------------------------

        if self._is_composer(agent_role):

            result = self._compose_final()

            return self._final_answer(result)

        # --------------------------------------------------------------
        # Fallback for unknown agent
        # --------------------------------------------------------------

        result = {
            "response": FALLBACK_MESSAGE,
            "source_type": "fallback",
            "grounded": False,
        }

        return self._final_answer(result)

    # ------------------------------------------------------------------
    # Agent identification
    # ------------------------------------------------------------------

    @staticmethod
    def _get_agent_role(
        kwargs: Dict[str, Any],
    ) -> str:
        """Extract the CrewAI agent role from call metadata."""

        agent = kwargs.get("from_agent")

        if agent is None:
            return ""

        role = getattr(agent, "role", "")

        return str(role)

    @staticmethod
    def _is_retrieval_agent(role: str) -> bool:
        return "retrieval agent" in role.lower()

    @staticmethod
    def _is_lookup_agent(role: str) -> bool:
        return "lookup agent" in role.lower()

    @staticmethod
    def _is_composer(role: str) -> bool:
        return "response composer" in role.lower()

    # ------------------------------------------------------------------
    # Task extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_task_text(
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Extract the complete CrewAI task text.

        The customer request is embedded inside the task description.
        """

        for message in reversed(messages):

            content = message.get("content", "")

            if content:
                return content

        return ""

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    @staticmethod
    def _run_rag(
        task_text: str,
        ) -> dict[str, Any]:
        """Actually execute the RAG policy tool."""

        from agents.crew import RAGLookupTool
        import ast

        query = MockLLM._extract_customer_request(
            task_text
        )

        tool = RAGLookupTool()

        raw_result = tool._run(query)

        # RAGLookupTool returns a Python-list representation.
        # Parse it into structured data for the composer.
        try:
            parsed_result = json.loads(raw_result)

            if isinstance(parsed_result, list):
                return {
                    "results": parsed_result,
                    "source_type": "policy",
                    "grounded": True,
                }

            if isinstance(parsed_result, dict):
                return parsed_result

        except json.JSONDecodeError:
            pass

        try:
            parsed_result = ast.literal_eval(raw_result)

            if isinstance(parsed_result, list):
                return {
                    "results": parsed_result,
                    "source_type": "policy",
                    "grounded": True,
                }

            if isinstance(parsed_result, dict):
                return parsed_result

        except (ValueError, SyntaxError):
            pass

        return {
            "response": raw_result,
            "source_type": "policy",
            "grounded": True,
        }

    # ------------------------------------------------------------------
    # Order lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _run_order_lookup(
        record_id: str,
    ) -> dict[str, Any]:
        """Actually execute the order lookup tool."""

        from agents.crew import OrderLookupTool

        tool = OrderLookupTool()

        raw_result = tool._run(record_id)

        try:
            return json.loads(raw_result)

        except json.JSONDecodeError:

            try:
                # Current OrderLookupTool historically returned
                # Python-dict text rather than JSON.
                import ast

                return ast.literal_eval(raw_result)

            except Exception:
                return {
                    "response": raw_result,
                    "source_type": "order",
                    "grounded": True,
                }

    # ------------------------------------------------------------------
    # Customer request extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_customer_request(
        task_text: str,
    ) -> str:
        """Extract text after 'Customer request:'."""

        match = re.search(
            r"Customer request:\s*(.+?)(?:\n|$)",
            task_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return task_text.strip()

    # ------------------------------------------------------------------
    # Composer
    # ------------------------------------------------------------------

    def _compose_final(self) -> dict[str, Any]:
        """
        Compose from actual tool results retained in this MockLLM instance.
        """

        rag_result = None
        order_result = None

        for item in reversed(self.tool_results):

            payload = item.get("payload")

            if not isinstance(payload, dict):
                continue

            if item.get("type") == "rag":
                rag_result = payload

            elif item.get("type") == "lookup":
                order_result = payload

        # --------------------------------------------------------------
        # Policy
        # --------------------------------------------------------------

        if rag_result:

            results = rag_result.get("results", [])

            if results:

                response_parts = []

                for result in results[:3]:

                    text = result.get("text")

                    if text:
                        response_parts.append(
                            str(text)
                        )

                if response_parts:

                    return {
                        "response": " ".join(
                            response_parts
                        ),
                        "source_type": "policy",
                        "grounded": True,
                    }

        # --------------------------------------------------------------
        # Order
        # --------------------------------------------------------------

        if order_result:

            if "status" in order_result:

                record_id = order_result.get(
                    "record_id",
                    "",
                )

                status = order_result.get(
                    "status",
                    "unknown",
                )

                return {
                    "response": (
                        f"Order {record_id} is currently "
                        f"'{status}'."
                    ),
                    "source_type": "order",
                    "grounded": True,
                }

        # --------------------------------------------------------------
        # Fallback
        # --------------------------------------------------------------

        return {
            "response": FALLBACK_MESSAGE,
            "source_type": "fallback",
            "grounded": False,
        }

    # ------------------------------------------------------------------
    # Final answer formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _final_answer(
        payload: dict[str, Any],
    ) -> str:

        return (
            "Final Answer: "
            + json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        )