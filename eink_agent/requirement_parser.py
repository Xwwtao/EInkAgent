"""Convert natural-language requests into structured device requirements."""

from openai import OpenAI

from eink_agent.requirements import DeviceRequirements

import json


SYSTEM_PROMPT = (
    "Extract only device search constraints explicitly stated by the user. "
    "Return a JSON object matching the supplied schema. "
    "Use CNY for prices, inches for screen sizes, and grams for weight. "
    "Set every unspecified field to null. "
    "Do not infer or invent preferences. "
    "Recognize Chinese number words and normalize them to numeric values. "
    "'最多花两千元' means max_price=2000. "
    "'重量不超过零点三公斤' means max_weight_g=300. "
    "Before returning JSON, check that every explicit supported constraint "
    "in the user's request is represented in the output. "
    "JSON schema: "
    + json.dumps(DeviceRequirements.model_json_schema())
    + "\nExample input: 预算不超过2000元。"
    + "\nExample JSON output: "
    + DeviceRequirements(max_price=2000).model_dump_json()
)

def parse_requirements(
    user_text: str,
    *,
    client: OpenAI,
    model: str,
) -> DeviceRequirements:
    """Parse one user request with an injected OpenAI client."""
    normalized_text = user_text.strip()
    if not normalized_text:
        raise ValueError("user_text must not be empty")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalized_text},
        ],
        response_format={"type": "json_object"},
        max_tokens=2048,
    )

    if not response.choices:
        raise RuntimeError("Model did not return structured requirements")

    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError("Model did not complete structured requirements")

    content = choice.message.content
    if not content or not content.strip():
        raise RuntimeError("Model did not return structured requirements")

    return DeviceRequirements.model_validate_json(content)
