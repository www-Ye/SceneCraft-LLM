"""
Unified LLM client supporting OpenAI and Anthropic APIs.
"""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Lightweight wrapper around LLM APIs for scene planning."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retry_attempts: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                logger.error("openai package not installed.")
                raise
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                logger.error("anthropic package not installed.")
                raise
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            temperature: Optional temperature override

        Returns:
            Response text from the LLM
        """
        temp = temperature if temperature is not None else self.temperature
        sys_prompt = system_prompt or (
            "You are an expert interior designer and 3D scene layout planner. "
            "Always respond with valid JSON when asked for structured output."
        )

        for attempt in range(self.retry_attempts):
            try:
                if self.provider == "openai":
                    return self._chat_openai(prompt, sys_prompt, temp)
                elif self.provider == "anthropic":
                    return self._chat_anthropic(prompt, sys_prompt, temp)
            except Exception as e:
                logger.warning(
                    f"LLM call attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def _chat_openai(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _chat_anthropic(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text

    def estimate_cost(self, num_scenes: int, avg_rounds: int = 5) -> dict:
        """Estimate API cost for generating N scenes."""
        # Rough token estimates per call
        avg_input_tokens = 1500
        avg_output_tokens = 1000
        calls_per_scene = avg_rounds  # understand + N assets + layout + refine

        total_input = num_scenes * calls_per_scene * avg_input_tokens
        total_output = num_scenes * calls_per_scene * avg_output_tokens

        # Approximate pricing (GPT-4o as of 2026)
        cost_per_1k_input = 0.0025
        cost_per_1k_output = 0.01

        return {
            "total_calls": num_scenes * calls_per_scene,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost_usd": (
                total_input / 1000 * cost_per_1k_input
                + total_output / 1000 * cost_per_1k_output
            ),
        }
