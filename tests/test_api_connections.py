"""Quick smoke tests to verify OpenAI and AWS Bedrock API connections work."""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def test_openai_whisper_connection():
    """Verify OpenAI API key works and Whisper model is accessible."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY not set in environment"

    client = OpenAI(api_key=api_key)

    # List models to verify the key works (cheap call, no audio needed)
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    assert "whisper-1" in model_ids, f"whisper-1 not found in available models: {model_ids[:10]}"
    print(f"  OpenAI OK — {len(models.data)} models available, whisper-1 found")


def test_bedrock_client_creation():
    """Verify AWS credentials are valid and Bedrock client can be created."""
    from notetaker.llm import get_client, _is_bedrock

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    assert region, "AWS_REGION not set in environment"
    assert region == "us-east-1", f"AWS_REGION is '{region}', expected 'us-east-1'"

    client = get_client(use_bedrock=True)
    assert _is_bedrock(client), "Client should be AnthropicBedrock"
    print(f"  Bedrock client created for region: {region}")


def test_bedrock_haiku_call():
    """Verify Haiku model works on Bedrock (cheapest possible call)."""
    from notetaker.llm import get_client, get_model_id, _is_bedrock

    client = get_client(use_bedrock=True)
    model_id = get_model_id("claude-haiku-4-5-20251001", _is_bedrock(client))

    response = client.messages.create(
        model=model_id,
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with just the word 'ok'"}],
    )
    text = response.content[0].text.strip().lower()
    assert "ok" in text, f"Unexpected response: {text}"
    print(f"  Bedrock Haiku OK — model: {model_id}, response: '{text}'")


def test_bedrock_sonnet_call():
    """Verify Sonnet model works on Bedrock."""
    from notetaker.llm import get_client, get_model_id, _is_bedrock

    client = get_client(use_bedrock=True)
    model_id = get_model_id("claude-sonnet-4-20250514", _is_bedrock(client))

    response = client.messages.create(
        model=model_id,
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with just the word 'ok'"}],
    )
    text = response.content[0].text.strip().lower()
    assert "ok" in text, f"Unexpected response: {text}"
    print(f"  Bedrock Sonnet OK — model: {model_id}, response: '{text}'")


def test_bedrock_opus_call():
    """Verify Opus 4.6 model works on Bedrock (used as orchestrator)."""
    from notetaker.llm import get_client, get_model_id, _is_bedrock

    client = get_client(use_bedrock=True)
    model_id = get_model_id("claude-opus-4-6", _is_bedrock(client))

    response = client.messages.create(
        model=model_id,
        max_tokens=10,
        messages=[{"role": "user", "content": "Reply with just the word 'ok'"}],
    )
    text = response.content[0].text.strip().lower()
    assert "ok" in text, f"Unexpected response: {text}"
    print(f"  Bedrock Opus OK — model: {model_id}, response: '{text}'")


def test_bedrock_tool_use():
    """Verify tool_use works on Bedrock (critical for the agent loop)."""
    from notetaker.llm import get_client, get_model_id, _is_bedrock

    client = get_client(use_bedrock=True)
    model_id = get_model_id("claude-haiku-4-5-20251001", _is_bedrock(client))

    test_tools = [
        {
            "name": "ping",
            "description": "A test tool that returns pong.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "A message."}
                },
                "required": ["message"],
            },
        }
    ]

    response = client.messages.create(
        model=model_id,
        max_tokens=100,
        tools=test_tools,
        messages=[{"role": "user", "content": "Use the ping tool with message 'hello'"}],
    )

    tool_calls = [b for b in response.content if b.type == "tool_use"]
    assert len(tool_calls) > 0, f"Expected tool_use, got: {[b.type for b in response.content]}"
    assert tool_calls[0].name == "ping"
    print(f"  Bedrock tool_use OK — tool called: {tool_calls[0].name}, input: {tool_calls[0].input}")


def test_ffmpeg_available():
    """Verify ffmpeg and ffprobe are installed."""
    import subprocess

    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    assert result.returncode == 0, "ffmpeg not found"
    version_line = result.stdout.split("\n")[0]
    print(f"  ffmpeg OK — {version_line}")

    result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
    assert result.returncode == 0, "ffprobe not found"
    version_line = result.stdout.split("\n")[0]
    print(f"  ffprobe OK — {version_line}")


if __name__ == "__main__":
    """Run all tests manually."""
    tests = [
        ("ffmpeg available", test_ffmpeg_available),
        ("OpenAI Whisper connection", test_openai_whisper_connection),
        ("Bedrock client creation", test_bedrock_client_creation),
        ("Bedrock Haiku call", test_bedrock_haiku_call),
        ("Bedrock Sonnet call", test_bedrock_sonnet_call),
        ("Bedrock Opus call", test_bedrock_opus_call),
        ("Bedrock tool_use", test_bedrock_tool_use),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[TEST] {name}...")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
