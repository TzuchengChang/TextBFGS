import json
import time
from typing import List, Dict

import requests


class QwenClient:
    """
    Minimal Qwen-compatible chat client used in the TextBFGS experiments.

    This client only implements the features required by our scripts:
    - streaming-style chunk reading from a server that returns `data: ...` lines
    - simple retry mechanism with configurable timeout and max_retries

    All endpoints and API keys are anonymized for the public release of this
    supplementary material. To actually run the code, you must:
    - set `api_key` to your own key, e.g. `api_key=\"Bearer YOUR_API_KEY_HERE\"`
    - set `base_url` to the base URL of your deployed model, e.g.
      `https://api.example.com/v1`
    """

    def __init__(
        self,
        model: str = "Qwen3-235B-A22B",
        api_key: str = "Bearer YOUR_API_KEY_HERE",
        base_url: str = "https://api.example.com/v1",
        timeout: int = 300,
        max_retries: int = 5,
        **kwargs,
    ):
        self.model: str = model
        self.api_key: str = api_key
        self.base_url: str = base_url.rstrip("/")
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        # default extra generation parameters (temperature, top_p, etc.)
        self.kwargs: Dict = kwargs

    def generate_message(
        self,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Send a chat completion request to the configured endpoint and return
        the concatenated content string.

        The server is expected to implement a streaming interface similar to
        `/chat/completions` endpoint, where each line starts with
        `data: ` and contains a JSON object with `choices[0].delta.content`.
        """
        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        # merge default kwargs with per-call overrides
        params = {**self.kwargs, **kwargs}
        params["stream"] = True

        request_body = {
            "model": self.model,
            "messages": messages,
        }

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url=f"{self.base_url}/chat/completions",
                    json={**request_body, **params},
                    headers={"Authorization": self.api_key},
                    timeout=self.timeout,
                    stream=True,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Request failed with status code {response.status_code}, "
                        f"message: {response.text}"
                    )

                content_parts: List[str] = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if not decoded.startswith("data: "):
                        continue

                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        # ignore malformed chunks
                        continue

                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content_parts.append(delta["content"])

                whole_text = "".join(content_parts)

                if "</think>" in whole_text:
                    final_content = whole_text.split("</think>")[-1].strip()
                else:
                    final_content = whole_text.strip()

                return final_content

            except Exception as exc:  # broad catch is intentional for robustness
                last_exception = exc
                if attempt == self.max_retries - 1:
                    break
                # simple exponential backoff
                time.sleep(self.timeout)

        if last_exception is not None:
            raise RuntimeError(
                f"Request failed after {self.max_retries} attempts: {last_exception}"
            )

        # Should be unreachable, but kept for type checkers
        raise RuntimeError("Unknown error: all retries failed with no exception captured.")


if __name__ == "__main__":
    # Minimal manual test (requires a real endpoint and API key).
    client = QwenClient()
    reply = client.generate_message(
        messages=[{"role": "user", "content": "Who are you?"}]
    )
    print(reply)

