from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# First API call with reasoning
response = client.chat.completions.create(
    model="x-ai/grok-4.1-fast:free",
    messages=[
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
    ],
    extra_body={"reasoning": {"enabled": True}},
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": response.content,
        "reasoning_details": response.reasoning_details,  # Pass back unmodified
    },
    {"role": "user", "content": "Are you sure? Think carefully."},
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
    model="x-ai/grok-4.1-fast:free",
    messages=messages,
    extra_body={"reasoning": {"enabled": True}},
)
