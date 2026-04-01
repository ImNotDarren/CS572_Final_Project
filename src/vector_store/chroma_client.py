"""ChromaDB vector store client for FNDDS food retrieval.

Handles remote ChromaDB servers that may sit behind a reverse proxy
with a URL path prefix (e.g. /chroma).
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from src.config import Settings

logger = logging.getLogger(__name__)


class _ChromaHTTP:
    """Lightweight ChromaDB v2 HTTP client that supports URL path prefixes."""

    def __init__(self, base_url: str, tenant: str, database: str) -> None:
        self._base = base_url.rstrip("/") + "/api/v2"
        self._tenant = tenant
        self._database = database
        self._http = httpx.Client(timeout=60.0)

    def _prefix(self) -> str:
        return f"{self._base}/tenants/{self._tenant}/databases/{self._database}"

    def get_collection(self, name: str) -> Dict[str, Any]:
        """Get collection metadata by name."""
        resp = self._http.get(f"{self._prefix()}/collections/{name}")
        resp.raise_for_status()
        return resp.json()

    def count(self, collection_id: str) -> int:
        resp = self._http.get(f"{self._prefix()}/collections/{collection_id}/count")
        resp.raise_for_status()
        return resp.json()

    def query(
        self,
        collection_id: str,
        query_embeddings: List[List[float]],
        n_results: int = 8,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query collection by embeddings."""
        body: Dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if include:
            body["include"] = include

        resp = self._http.post(
            f"{self._prefix()}/collections/{collection_id}/query",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


class ChromaRetriever:
    """Retrieves food entries from the FNDDS ChromaDB collection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._openai = OpenAI(api_key=settings.openai_api_key)

        logger.info("Connecting to ChromaDB at %s", settings.chroma_url)
        self._client = _ChromaHTTP(
            base_url=settings.chroma_url,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )

        # Get collection metadata
        coll_info = self._client.get_collection(settings.chroma_collection)
        self._collection_id = coll_info["id"]
        count = self._client.count(self._collection_id)
        logger.info(
            "Connected to collection '%s' (id=%s) with %d entries",
            settings.chroma_collection,
            self._collection_id,
            count,
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI."""
        response = self._openai.embeddings.create(
            model=self._settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        """Search for similar food entries.

        Returns list of dicts with 'food_code' and 'description'.
        """
        embeddings = self._embed([query])
        results = self._client.query(
            collection_id=self._collection_id,
            query_embeddings=embeddings,
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        entries: List[Dict] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                document = results["documents"][0][i] if results.get("documents") else ""
                food_code = metadata.get("Food code", doc_id)
                entries.append({
                    "food_code": str(food_code),
                    "description": document,
                })
        return entries

    def multi_search(
        self, queries: List[str], top_k: int = 8
    ) -> List[Dict]:
        """Search multiple queries, deduplicate by food code."""
        seen_codes: set = set()
        all_entries: List[Dict] = []

        for query in queries:
            try:
                results = self.search(query, top_k=top_k)
                for entry in results:
                    code = entry["food_code"]
                    if code not in seen_codes:
                        seen_codes.add(code)
                        all_entries.append(entry)
            except Exception as e:
                logger.warning("Search failed for query '%s': %s", query[:50], e)

        logger.info(
            "Multi-search: %d queries -> %d unique entries",
            len(queries),
            len(all_entries),
        )
        return all_entries
