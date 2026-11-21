"""
Hybrid retrieval: BM25 (lexical) + vector search (semantic), merged with
weighted Reciprocal Rank Fusion (RRF).

Why hybrid?
    - BM25 is excellent at exact tokens: SKU codes, error codes, IDs
      ("SKU-7742X", "E_CONN_REFUSED").
    - Vector search is excellent at meaning: "can't connect" ~ "connectivity issues".
    Each one fails where the other succeeds, so we fuse their rankings.

Requirements:
    pip install langchain-core langchain-chroma langchain-openai \
                langchain-community rank_bm25
    export OPENAI_API_KEY=...
"""

import re
from typing import Callable, List, Sequence

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def hybrid_retrieve(
    query: str,
    retrievers: Sequence[BaseRetriever],
    weights: Sequence[float],
    k: int = 3,
    rrf_k: int = 60,
) -> List[Document]:
    """Combine several retrievers with weighted Reciprocal Rank Fusion (RRF).

    For each retriever, a document at rank ``r`` (1 = best) earns
    ``weight / (rrf_k + r)``. Scores are summed across retrievers, so a
    document found by both lists rises to the top.

    Args:
        query: User query.
        retrievers: Retrievers to query (each must support ``.invoke``).
        weights: One weight per retriever (higher = more influence).
        k: Number of documents to return.
        rrf_k: RRF smoothing constant (60 is the standard value). Larger
            values flatten the difference between top ranks.

    Returns:
        The top-``k`` documents sorted by fused score (descending).
    """
    if len(retrievers) != len(weights):
        raise ValueError("`retrievers` and `weights` must have the same length.")

    # page_content -> (cumulative score, document)
    doc_scores: dict[str, tuple[float, Document]] = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        # RRF ranks start at 1 (not 0).
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content  # used to deduplicate across retrievers
            rrf_score = weight * (1.0 / (rrf_k + rank))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc_scores[key][1])
            else:
                doc_scores[key] = (rrf_score, doc)

    ranked = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


# --------------------------------------------------------------------------- #
# BM25 tokenizer
# --------------------------------------------------------------------------- #
def code_friendly_tokenize(text: str) -> List[str]:
    """Lowercase tokenizer that keeps identifiers intact.

    The default BM25 tokenizer splits on whitespace only, so "specifications:"
    would not match "specifications". Here punctuation is stripped but
    '-', '_' and '.' inside a token are kept, so "SKU-7742X",
    "E_CONN_REFUSED" and "192.168.1.1" stay whole.
    """
    return re.findall(r"[\w][\w\-\.]*[\w]|[\w]", text.lower())


# --------------------------------------------------------------------------- #
# Hybrid retriever
# --------------------------------------------------------------------------- #
class HybridRetriever:
    """BM25 + vector (Chroma) retriever fused with weighted RRF."""

    def __init__(
        self,
        documents: List[Document],
        bm25_weight: float = 0.5,
        k: int = 4,
        fetch_k: int | None = None,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "hybrid_search",
        tokenizer: Callable[[str], List[str]] = code_friendly_tokenize,
    ):
        """
        Args:
            documents: Initial corpus.
            bm25_weight: Weight of BM25 in [0, 1]; vector weight is 1 - this.
            k: Number of documents returned by ``search``.
            fetch_k: Candidates pulled from each retriever before fusion
                (default ``2 * k``). A larger pool gives RRF more to work with.
            embedding_model: OpenAI embedding model name.
            collection_name: Chroma collection name.
            tokenizer: Tokenizer used by BM25.
        """
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be between 0 and 1.")

        self.k = k
        self.fetch_k = fetch_k or 2 * k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1.0 - bm25_weight
        self.tokenizer = tokenizer

        # Keep our own copy of the corpus (with metadata) to rebuild BM25.
        self.documents: List[Document] = list(documents)

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = Chroma.from_documents(
            self.documents, self.embeddings, collection_name=collection_name
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.fetch_k}
        )
        self.bm25_retriever = self._build_bm25()

    def _build_bm25(self) -> BM25Retriever:
        return BM25Retriever.from_documents(
            self.documents, k=self.fetch_k, preprocess_func=self.tokenizer
        )

    def search(self, query: str) -> List[Document]:
        """Run hybrid search and return the top-``k`` documents."""
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both indexes.

        Chroma supports incremental adds; BM25 does not, so it is rebuilt from
        the full corpus (metadata preserved).
        """
        self.vectorstore.add_documents(documents)
        self.documents.extend(documents)
        self.bm25_retriever = self._build_bm25()


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
SAMPLE_DOCUMENTS = [
    Document(
        page_content="Product SKU-7742X specifications: Weight 2.5kg, Dimensions 30x20x10cm.",
        metadata={"type": "specs"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]


if __name__ == "__main__":
    retriever = HybridRetriever(SAMPLE_DOCUMENTS, bm25_weight=0.5, k=4)

    for query in ["SKU-7742X specifications", "my internet is not working"]:
        print(f"\nQuery: {query}")
        for doc in retriever.search(query):
            print(f"  [{doc.metadata.get('type')}] {doc.page_content[:100]}")