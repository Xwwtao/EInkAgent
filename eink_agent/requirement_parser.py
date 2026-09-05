"""Convert natural-language requests into structured device requirements."""

from openai import OpenAI

from eink_agent.requirements import DeviceRequirements

SYSTEM_PROMPT = (
    "Extract only device search constraints explicitly stated by the user. "
    "User CNY for prices, inches for screen sizes, and grams for weight. "
    "Set every unspecified field to null. "
    "Do not infer or invent preferences."
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

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": normalized_text,
            },
        ],
        text_format=DeviceRequirements,
    )

    requirements = response.output_parsed
    if requirements is None:
        raise RuntimeError("Model did not return structured requirements")

    return requirements
