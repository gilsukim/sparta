#!/usr/bin/env python3
"""Send a prompt to all available LLMs in parallel and combine their results."""

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Optional

import anthropic

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class LLMResult:
    provider: str
    model: str
    response: Optional[str]
    error: Optional[str] = None


async def query_claude(prompt: str) -> LLMResult:
    model = "claude-opus-4-7"
    try:
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model=model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return LLMResult(provider="Anthropic", model=model, response=text)
    except Exception as exc:
        return LLMResult(provider="Anthropic", model=model, response=None, error=str(exc))


async def query_openai(prompt: str) -> LLMResult:
    model = "gpt-4o"
    if not OPENAI_AVAILABLE:
        return LLMResult(provider="OpenAI", model=model, response=None,
                         error="openai package not installed — run: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return LLMResult(provider="OpenAI", model=model, response=None,
                         error="OPENAI_API_KEY environment variable not set")
    try:
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return LLMResult(provider="OpenAI", model=model, response=text)
    except Exception as exc:
        return LLMResult(provider="OpenAI", model=model, response=None, error=str(exc))


async def query_gemini(prompt: str) -> LLMResult:
    model = "gemini-1.5-pro"
    if not GEMINI_AVAILABLE:
        return LLMResult(provider="Google", model=model, response=None,
                         error="google-generativeai package not installed — run: pip install google-generativeai")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return LLMResult(provider="Google", model=model, response=None,
                         error="GOOGLE_API_KEY environment variable not set")
    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(model)
        # google-generativeai is sync; run in a thread pool to avoid blocking
        resp = await asyncio.to_thread(gemini.generate_content, prompt)
        return LLMResult(provider="Google", model=model, response=resp.text)
    except Exception as exc:
        return LLMResult(provider="Google", model=model, response=None, error=str(exc))


async def query_all(prompt: str) -> list[LLMResult]:
    """Query all LLMs concurrently."""
    return list(await asyncio.gather(
        query_claude(prompt),
        query_openai(prompt),
        query_gemini(prompt),
    ))


def display(prompt: str, results: list[LLMResult]) -> None:
    sep = "=" * 64
    thin = "-" * 64
    print(f"\n{sep}")
    print(f"  PROMPT: {prompt}")
    print(f"{sep}\n")

    for result in results:
        print(thin)
        print(f"  [{result.provider}]  {result.model}")
        print(thin)
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            # Indent each line of the response for readability
            for line in (result.response or "").splitlines():
                print(f"  {line}")
        print()

    ok = sum(1 for r in results if r.error is None)
    print(sep)
    print(f"  {ok}/{len(results)} LLMs responded successfully")
    print(f"{sep}\n")


async def main() -> None:
    if len(sys.argv) >= 2:
        prompt = " ".join(sys.argv[1:]).strip()
    else:
        prompt = input("Enter your prompt: ").strip()

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    print(f"\nQuerying {len([query_claude, query_openai, query_gemini])} LLMs in parallel…")
    results = await query_all(prompt)
    display(prompt, results)


if __name__ == "__main__":
    asyncio.run(main())
