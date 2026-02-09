from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from retrieval.base import SearchHit, RetrievalQualityFlag

@dataclass
class FusedCandidate:
    source_type: str
    chunk_id: str
    chunk_type: str
    text: str
    metadata: dict[str, Any]
    input_file: str
    
    rrf_score: float = 0.0
    
    sparse_rank: int | None = None
    sparse_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    vector_distance: float | None = None
    
    token_count: int | None = None


class RRFFuser:
    def __init__(
        self,
        rrf_rank_constant: int = 60,
        sparse_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> None:
        self.rrf_rank_constant = rrf_rank_constant
        self.sparse_weight = sparse_weight
        self.vector_weight = vector_weight

    def fuse(
        self,
        sparse_hits: Sequence[SearchHit],
        vector_hits: Sequence[SearchHit],
        quality_flags: list[RetrievalQualityFlag] | None = None,
    ) -> list[FusedCandidate]:
        if quality_flags is None:
            quality_flags = []

        fused: dict[tuple[str, str], FusedCandidate] = {}

        self._ingest(
            hits=sparse_hits,
            backend="sparse",
            weight=self.sparse_weight,
            fused=fused,
        )
        self._ingest(
            hits=vector_hits,
            backend="vector",
            weight=self.vector_weight,
            fused=fused,
        )

        # Sort candidates
        # Sort key: (-rrf_score, min_rank, source_type, chunk_id)
        # min_rank is the best rank from either backend (smaller is better)
        
        candidates = list(fused.values())
        candidates.sort(key=self._sort_key)
        
        return candidates

    def _sort_key(self, item: FusedCandidate) -> tuple[float, int, str, str]:
        ranks = [r for r in (item.sparse_rank, item.vector_rank) if r is not None]
        min_rank = min(ranks) if ranks else 10**9
        return (-item.rrf_score, min_rank, item.source_type, item.chunk_id)

    def _ingest(
        self,
        hits: Sequence[SearchHit],
        backend: str,
        weight: float,
        fused: dict[tuple[str, str], FusedCandidate],
    ) -> None:
        for idx, hit in enumerate(hits, start=1):
            # idx is the rank (1-based)
            rank = idx
            
            # Key: (source_type, chunk_id)
            # SearchHit has source_type and chunk_id
            key = (hit.source_type, hit.chunk_id)
            
            candidate = fused.get(key)
            if candidate is None:
                # Create new candidate
                # Extract fields from hit
                # SearchHit has content (text), metadata
                # We expect metadata to optionally contain token_count, input_file, chunk_type?
                # Or we rely on SearchHit fields?
                # SearchHit does NOT have chunk_type or input_file explicitly defined in base.py? 
                # Wait, step 508: SearchHit has chunk_id, document_id, score, content, metadata, source_type.
                # It does NOT have chunk_type, input_file, token_count.
                # So we must get them from metadata.
                
                meta = hit.metadata or {}
                chunk_type = str(meta.get("chunk_type") or "text")
                input_file = str(meta.get("input_file") or hit.document_id) # fallback?
                token_count = meta.get("token_count")
                if token_count is not None:
                    try:
                        token_count = int(token_count)
                    except (ValueError, TypeError):
                        token_count = None
                
                candidate = FusedCandidate(
                    source_type=hit.source_type,
                    chunk_id=hit.chunk_id,
                    chunk_type=chunk_type,
                    text=hit.content,
                    metadata=meta,
                    input_file=input_file,
                    token_count=token_count
                )
                fused[key] = candidate
            
            # Update RRF score
            contrib = weight * (1.0 / (self.rrf_rank_constant + rank))
            candidate.rrf_score += contrib
            
            # Update backend-specific info
            if backend == "sparse":
                if candidate.sparse_rank is None: # take best rank if duplicates?
                    candidate.sparse_rank = rank
                    candidate.sparse_score = hit.score
            elif backend == "vector":
                if candidate.vector_rank is None:
                    candidate.vector_rank = rank
                    candidate.vector_score = hit.score
                    # vector_distance not in SearchHit directly, maybe in metadata?
                    # For now leave None or check metadata
                    candidate.vector_distance = hit.metadata.get("distance")
