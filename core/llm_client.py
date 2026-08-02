"""
LLM Client with multi-URL load balancing for vLLM inference servers.

Features:
- Round-robin load balancing across multiple vLLM API endpoints
- Automatic retries with exponential backoff
- Thread-safe URL dispatching
- OpenAI-compatible API format
"""

import time
import logging
import threading
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    api_urls: List[str]  # List of vLLM API base URLs
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.8
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0


class URLDispatcher:
    """
    Thread-safe round-robin URL dispatcher.
    Distributes requests evenly across multiple vLLM endpoints.
    """

    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError("At least one URL is required")
        self.urls = urls
        self._counter = 0
        self._lock = threading.Lock()

    def get_next(self) -> str:
        """Get the next URL in round-robin fashion."""
        with self._lock:
            url = self.urls[self._counter % len(self.urls)]
            self._counter += 1
            return url

    def __len__(self) -> int:
        return len(self.urls)


class LLMClient:
    """
    LLM client with multi-URL load balancing.

    Usage:
        client = LLMClient(
            api_urls=["http://localhost:8001/v1", "http://localhost:8002/v1"],
            model="YOUR_MODEL"
        )
        response = client.chat("Hello!")
    """

    def __init__(
        self,
        api_urls: List[str],
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.8,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if isinstance(api_urls, str):
            api_urls = [api_urls]

        self.config = LLMConfig(
            api_urls=api_urls,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.dispatcher = URLDispatcher(api_urls)
        # Pre-create clients for each URL for connection reuse
        self._clients: Dict[str, OpenAI] = {}
        for url in api_urls:
            self._clients[url] = OpenAI(
                base_url=url,
                api_key=api_key,
                timeout=timeout,
            )

    def _get_client(self) -> OpenAI:
        """Get the next client via round-robin."""
        url = self.dispatcher.get_next()
        return self._clients[url]

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send a chat request to one of the vLLM endpoints.

        Args:
            prompt: User prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            top_p: Override default top_p
            system_prompt: Optional system prompt

        Returns:
            Response text or None if all retries failed
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Optional[str]:
        """
        Send a chat request with full messages list.

        Returns:
            Response text or None if all retries failed
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        tp = top_p if top_p is not None else self.config.top_p

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    top_p=tp,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries}), "
                        f"retrying in {sleep_time:.1f}s: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"LLM call failed after {self.config.max_retries} attempts: {last_error}"
                    )

        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """Create LLMClient from a config dictionary."""
        return cls(
            api_urls=config["api_urls"],
            model=config["model"],
            api_key=config.get("api_key", "EMPTY"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4096),
            top_p=config.get("top_p", 0.8),
            timeout=config.get("timeout", 120),
            max_retries=config.get("max_retries", 3),
            retry_delay=config.get("retry_delay", 1.0),
        )
