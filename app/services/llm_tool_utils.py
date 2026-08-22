"""
app/services/llm_tool_utils.py

Structured LLM output via OpenAI's tool-calling API (forced tool_choice +
strict JSON schema) instead of response_format={"type": "json_object"} +
a prompt-described JSON shape.

Why this is more reliable: the previous pattern asks the model to follow
a JSON shape described in plain-English prompt instructions, then
validates the result client-side with pydantic AFTER the fact — a model
that ignores an instruction just produces a ValidationError. Strict
tool-calling instead gives OpenAI's API itself the schema up front and
constrains generation to match it, so schema-shape failures (missing
fields, wrong types, wrapping the JSON in markdown despite being told not
to) become far less likely. It does NOT replace the need for the
hallucination/validity guards already used throughout this pipeline
(e.g. checking a returned column name actually exists in the schema) —
those are about the CONTENT of the answer, not just whether it parses.

LIMITATION: strict mode requires a fixed, known-ahead-of-time set of
property names. It does not reliably support dynamic/open-ended object
keys (e.g. a dict keyed by group names decided at runtime, as used in
model_selector's output). For that shape, keep the existing
response_format=json_object pattern, or restructure the output as a
fixed-shape list of {key, value} objects instead of a dict.
"""

from __future__ import annotations

import json
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMToolCallError(Exception):
    """Raised when the model doesn't call the tool as required, or the
    arguments it returns don't validate against the expected schema."""
    pass


def _patch_strict(node: dict, defs: dict) -> None:
    """
    Recursively rewrites a pydantic-generated JSON schema to satisfy
    OpenAI strict mode's requirements: every object needs
    "additionalProperties": false, and every property must be listed in
    "required" (optionality is expressed through the type itself, e.g.
    Optional[str] already becomes {"anyOf": [{"type": "string"}, {"type": "null"}]}
    via pydantic, which strict mode does support).
    """
    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        props = node.get("properties", {})
        node["required"] = list(props.keys())
        for prop_schema in props.values():
            _resolve_and_patch(prop_schema, defs)
    if node.get("type") == "array" and "items" in node:
        _resolve_and_patch(node["items"], defs)


def _resolve_and_patch(schema: dict, defs: dict) -> None:
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        _patch_strict(defs[ref_name], defs)
    elif "anyOf" in schema:
        for sub in schema["anyOf"]:
            _resolve_and_patch(sub, defs)
    else:
        _patch_strict(schema, defs)


def build_strict_schema(model: Type[BaseModel]) -> dict:
    """Converts a pydantic model into an OpenAI-strict-mode-compatible
    JSON schema. Exposed separately from call_llm_tool so it can be unit
    tested / inspected without making a real API call."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    _patch_strict(schema, defs)
    return schema


def call_llm_tool(
    client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    tool_name: str,
    tool_description: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> T:
    """
    Makes a structured LLM call via forced tool-calling + strict schema,
    and returns a validated instance of response_model.

    Raises LLMToolCallError (not a bare exception) if the model doesn't
    call the tool, or if the arguments don't validate — callers should
    catch this the same way they previously caught
    (json.JSONDecodeError, ValidationError) from the old pattern.
    """
    schema = build_strict_schema(response_model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": schema,
                    "strict": True,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )

    message = response.choices[0].message
    if not message.tool_calls:
        raise LLMToolCallError(
            f"Model did not call the '{tool_name}' tool as required. "
            f"Raw message content: {message.content!r}"
        )

    raw_args = message.tool_calls[0].function.arguments
    try:
        return response_model.model_validate(json.loads(raw_args))
    except Exception as e:
        raise LLMToolCallError(f"Tool call arguments failed validation: {e}\nRaw arguments: {raw_args}")