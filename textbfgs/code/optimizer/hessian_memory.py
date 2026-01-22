"""Hessian-Proxy Knowledge Base (HPKB) for T-BFGS (anonymized snapshot).

This module stores and retrieves optimization trajectories:
    - textual gradients
    - abstract operators
    - before/after code (optional)

The concrete embedding endpoint and API key are intentionally anonymized.
To run HPKB in practice, you must configure your own embedding service.
"""

import os
import time
from typing import Dict, List, Optional

import chromadb
import httpx
from chromadb.config import Settings

from textgrad import logger


class HessianProxyKB:
    """
    Hessian-Proxy Knowledge Base for storing and retrieving optimization trajectories.

    Each trajectory stores:
    - gradient_text : textual feedback (used as the primary search key)
    - operator_text : high-level rule describing the fix
    - old_val       : pre-optimization code
    - new_val       : post-optimization code

    The gradient text is embedded via an external embedding API and used to
    query a ChromaDB collection for nearest neighbours.
    """

    def __init__(
        self,
        embedding_api_url: str = "https://api.example.com/embeddings",
        api_key: str = "YOUR_EMBEDDING_API_KEY_HERE",
        embedding_model: str = "Qwen3-Embedding-8B",
        kb_path: str = "./t_bfgs_kb",
        collection_name: str = "hessian_proxy_kb",
    ):
        self.embedding_api_url = embedding_api_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.kb_path = kb_path

        os.makedirs(kb_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=kb_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Hessian-Proxy Knowledge Base for T-BFGS"},
        )

        logger.info(
            "HessianProxyKB initialized with %d existing entries", self.collection.count()
        )

    def _create_operator_description(self, old_val: str, new_val: str) -> str:
        """
        Create a concise operator description from the transformation.
        This is a lightweight heuristic summarizer used when no explicit
        <OPERATOR> section is provided by the model.
        """
        old_lines = old_val.split("\n")
        new_lines = new_val.split("\n")

        if len(old_lines) != len(new_lines):
            return f"Code structure changed: {len(old_lines)} lines -> {len(new_lines)} lines"

        old_lower = old_val.lower()
        new_lower = new_val.lower()

        if "if" not in old_lower and "if" in new_lower:
            return "Added conditional check"
        if "try" not in old_lower and "try" in new_lower:
            return "Added error handling"
        if "return" in old_lower and "return" in new_lower:
            old_returns = [line for line in old_lines if "return" in line.lower()]
            new_returns = [line for line in new_lines if "return" in line.lower()]
            if old_returns != new_returns:
                return "Modified return logic"

        return "Code optimization applied"

    def _encode(
        self,
        text: str,
        max_retries: int = 3,
        timeout: Optional[float] = None,
    ) -> List[float]:
        """
        Encode text into an embedding vector via the configured embedding API.

        The exact request/response format depends on your deployment; here we
        assume an JSON interface returning `data[0].embedding`.
        """
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    self.embedding_api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": [text],
                        "model": self.embedding_model,  # configurable embedding model name
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()

                if "data" in result and result["data"]:
                    embedding = result["data"][0].get("embedding", result["data"][0])
                    return list(map(float, embedding))

                raise ValueError(f"Unexpected embedding API response format: {result}")

            except (httpx.TimeoutException, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Embedding API timeout (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        exc,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Failed to encode text after %d attempts: %s", max_retries, exc
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Embedding API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        exc,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Failed to encode text after %d attempts: %s", max_retries, exc
                    )

        logger.warning(
            "Using zero-vector fallback for embedding due to previous error: %s", last_error
        )
        # Dimension is implementation-dependent; 1024 is a safe placeholder.
        return [0.0] * 1024

    def add_trace(
        self,
        gradient_text: str,
        old_val: str,
        new_val: str,
        domain: str = "General",
        operator_text: Optional[str] = None,
    ) -> None:
        """
        Add a successful optimization trajectory to the knowledge base.
        """
        try:
            embedding = self._encode(gradient_text)

            if operator_text and operator_text.strip():
                operator_description = operator_text.strip()
                logger.debug(
                    "Using model-generated operator: %s...", operator_description[:200]
                )
            else:
                operator_description = self._create_operator_description(old_val, new_val)
                logger.debug(
                    "Using auto-generated operator: %s...", operator_description[:200]
                )

            import hashlib

            trace_id = hashlib.md5(
                f"{gradient_text}_{old_val}_{new_val}_{domain}".encode("utf-8")
            ).hexdigest()

            self.collection.add(
                embeddings=[embedding],
                ids=[trace_id],
                metadatas=[
                    {
                        "gradient_summary": gradient_text[:500]
                        if len(gradient_text) > 500
                        else gradient_text,
                        "operator_summary": operator_description[:500]
                        if len(operator_description) > 500
                        else operator_description,
                        "domain": domain,
                        "gradient_length": str(len(gradient_text)),
                        "old_val_length": str(len(old_val)),
                        "new_val_length": str(len(new_val)),
                    }
                ],
                documents=[
                    "GRADIENT:\n"
                    + gradient_text
                    + "\n\nOPERATOR:\n"
                    + operator_description
                    + "\n\nOLD_VALUE:\n"
                    + old_val
                    + "\n\nNEW_VALUE:\n"
                    + new_val
                ],
            )

            logger.info("Added trace to KB: domain=%s, trace_id=%s", domain, trace_id[:8])

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to add trace to KB: %s", exc)

    def retrieve_inverse_hessian(
        self,
        current_gradient: str,
        domain: str = "General",
        top_k: int = 3,
        include_code_examples: bool = False,
        current_old_val: Optional[str] = None,
    ) -> str:
        """
        Retrieve historical optimization patterns similar to the current situation.
        """
        try:
            if self.collection.count() == 0:
                return (
                    "### Optimization Manifold (Historical Gradients & Fixes):\n"
                    "No historical trajectories available yet.\n"
                )

            if include_code_examples and current_old_val is not None:
                query_text = current_old_val
            else:
                query_text = current_gradient

            query_embedding = self._encode(query_text)

            where_clause = None
            if domain != "General":
                where_clause = {"domain": domain}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            context = "### Optimization Manifold (Historical Gradients & Fixes):\n"
            context += "The following are successful optimization patterns from similar situations:\n\n"

            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][i]
                    document = (
                        results["documents"][0][i]
                        if results["documents"] and results["documents"][0]
                        else None
                    )
                    result_domain = metadata.get("domain", "Unknown")
                    distance = (
                        results["distances"][0][i]
                        if results["distances"] and results["distances"][0]
                        else None
                    )

                    if document:
                        parts: Dict[str, str] = {}
                        current_key: Optional[str] = None
                        current_value: List[str] = []

                        for line in document.split("\n"):
                            if line.startswith("GRADIENT:"):
                                if current_key:
                                    parts[current_key] = "\n".join(current_value).strip()
                                current_key = "gradient"
                                current_value = []
                            elif line.startswith("OPERATOR:"):
                                if current_key:
                                    parts[current_key] = "\n".join(current_value).strip()
                                current_key = "operator"
                                current_value = []
                            elif line.startswith("OLD_VALUE:"):
                                if current_key:
                                    parts[current_key] = "\n".join(current_value).strip()
                                current_key = "old_value"
                                current_value = []
                            elif line.startswith("NEW_VALUE:"):
                                if current_key:
                                    parts[current_key] = "\n".join(current_value).strip()
                                current_key = "new_value"
                                current_value = []
                            elif current_key:
                                current_value.append(line)

                        if current_key:
                            parts[current_key] = "\n".join(current_value).strip()

                        gradient_text = parts.get("gradient", "")
                        operator_text = parts.get("operator", "")
                        old_val_text = parts.get("old_value", "")
                        new_val_text = parts.get("new_value", "")
                    else:
                        gradient_text = metadata.get("gradient_summary", "")
                        operator_text = metadata.get("operator_summary", "")
                        old_val_text = ""
                        new_val_text = ""

                    context += f"#### Pattern {i + 1} (Domain: {result_domain}"
                    if distance is not None:
                        context += f", Similarity: {1 - distance:.3f}"
                    context += ")\n\n"

                    if include_code_examples:
                        if old_val_text:
                            context += f"**Previous Value:**\n```\n{old_val_text}\n```\n\n"
                        if new_val_text:
                            context += (
                                "**Optimized Value (Complete Fix Pattern):**\n"
                                f"```\n{new_val_text}\n```\n\n"
                            )
                    else:
                        context += f"**Feedback Received:**\n{gradient_text}\n\n"
                        clean_operator = operator_text
                        if operator_text.startswith("Transformation applied to resolve:"):
                            clean_operator = operator_text[
                                len("Transformation applied to resolve:") :
                            ].strip()
                        context += f"**Transformation Applied:**\n{clean_operator}\n\n"

                    context += "---\n\n"
            else:
                context += "No similar historical trajectories found. This is a new optimization scenario.\n"

            return context

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to retrieve from KB: %s", exc)
            return (
                "### Optimization Manifold (Historical Gradients & Fixes):\n"
                "Error retrieving historical trajectories.\n"
            )

    def get_stats(self) -> Dict:
        """Return basic statistics about the knowledge base."""
        count = self.collection.count()
        all_metadata = self.collection.get()["metadatas"]
        domain_counts: Dict[str, int] = {}
        for meta in all_metadata:
            domain = meta.get("domain", "Unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_traces": count,
            "domains": domain_counts,
            "kb_path": self.kb_path,
            "collection_name": self.collection.name,
        }

    @staticmethod
    def list_collections(kb_path: str) -> List[str]:
        """List all collections available under a given KB path."""
        try:
            client = chromadb.PersistentClient(
                path=kb_path, settings=Settings(anonymized_telemetry=False)
            )
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to list collections: %s", exc)
            return []

    def clear_collection(self) -> None:
        """Remove all entries from the current collection."""
        try:
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"description": "Hessian-Proxy Knowledge Base for T-BFGS"},
            )
            logger.info("Cleared collection: %s", self.collection.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to clear collection: %s", exc)

