from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

import tiktoken

from generator.config import (
    GeneratorRuntimeConfig,
    GeneratorSettings,
    get_generator_config,
    get_generator_settings,
)
from generator.models import (
    GenerateRequest,
    GenerateResponse,
    GenerationQualityFlag,
    LLMStructuredOutput,
    RetrievalHit,
    RetrievalQualityFlag,
    RetrievalSearchResponse,
    UsedChunk,
)


class GeneratorBackendUnavailableError(RuntimeError):
    """Raised when required external services are not reachable."""


class TokenEstimator:
    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text or ""))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        encoded = self._encoding.encode(text or "")
        if len(encoded) <= max_tokens:
            return text
        return self._encoding.decode(encoded[:max_tokens]).strip()


class GeneratorService:
    def __init__(
        self,
        *,
        settings: GeneratorSettings | None = None,
        runtime_config: GeneratorRuntimeConfig | None = None,
    ) -> None:
        self.settings = settings or get_generator_settings()
        self.runtime_config = runtime_config or get_generator_config(self.settings)
        self._token_estimator = TokenEstimator()
        
        from generator.llm_client import LLMClient
        self.llm_client = LLMClient(self.runtime_config)

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        timeout_seconds: float,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        # Simple local implementation for retrieval calls
        # We could also use self.llm_client._request_json if we made it public or shared utility
        # For now, keeping it here to avoid breaking retrieval calls
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request_obj = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
                raw_body = response.read()
                if not raw_body:
                    return int(response.status), {}, None
                body = json.loads(raw_body.decode("utf-8"))
                if not isinstance(body, dict):
                    return int(response.status), None, "Response is not a JSON object."
                return int(response.status), body, None
        except urllib.error.HTTPError as exc:
            return int(exc.code), None, str(exc)
        except urllib.error.URLError as exc:
            return None, None, str(exc.reason)
        except Exception as exc:
            return None, None, str(exc)

    def _call_retrieval_search(self, request: GenerateRequest, document_id: str, top_k: int) -> RetrievalSearchResponse:
        payload: dict[str, Any] = {
            "query": request.query.strip(),
            "document_id": document_id,
            "top_k": top_k,
            "rebuild_if_stale": (
                request.retrieval_rebuild_if_stale
                if request.retrieval_rebuild_if_stale is not None
                else self.runtime_config.retrieval_rebuild_if_stale_default
            ),
        }
        if request.source_types:
            payload["source_types"] = request.source_types

        status, body, error = self._request_json(
            method="POST",
            url=f"{self.runtime_config.retrieval_base_url}/search",
            timeout_seconds=self.runtime_config.retrieval_timeout_seconds,
            payload=payload,
        )

        if status is None:
            raise GeneratorBackendUnavailableError(
                f"Retrieval backend unavailable: {error or 'unknown error'}"
            )
        if status < 200 or status >= 300 or body is None:
            raise GeneratorBackendUnavailableError(
                f"Retrieval backend request failed ({status}): {error or 'unknown error'}"
            )

        try:
            return RetrievalSearchResponse.model_validate(body)
        except Exception as exc:
            raise GeneratorBackendUnavailableError(
                f"Retrieval response validation failed: {exc}"
            ) from exc

    def _call_ollama_chat(
        self,
        *,
        model: str,
        temperature: float,
        total_budget_tokens: int,
        max_answer_tokens: int,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[LLMStructuredOutput | None, str, str | None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return self.llm_client.chat_structured(
            model=model,
            messages=messages,
            temperature=temperature,
            num_ctx=total_budget_tokens,
            num_predict=max_answer_tokens,
            response_model=LLMStructuredOutput
        )

    def _call_retrieval_ready(self) -> tuple[bool, dict[str, Any]]:
        status, body, error = self._request_json(
            method="GET",
            url=f"{self.runtime_config.retrieval_base_url}/ready",
            timeout_seconds=self.runtime_config.retrieval_timeout_seconds,
        )
        if status is None or status < 200 or status >= 300 or body is None:
            return False, {
                "available": False,
                "status_code": status,
                "error": error,
            }
        return True, {
            "available": True,
            "status_code": status,
            "ready_payload": body,
        }

    def _call_ollama_ready(self) -> tuple[bool, dict[str, Any]]:
        status, body, error = self._request_json(
            method="GET",
            url=f"{self.runtime_config.ollama_base_url}/api/tags",
            timeout_seconds=min(self.runtime_config.ollama_timeout_seconds, 15.0),
        )
        if status is None or status < 200 or status >= 300 or body is None:
            return False, {
                "available": False,
                "status_code": status,
                "error": error,
            }

        model_names: list[str] = []
        raw_models = body.get("models")
        if isinstance(raw_models, list):
            for entry in raw_models:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if name:
                    model_names.append(name)

        return True, {
            "available": True,
            "status_code": status,
            "model_count": len(model_names),
            "models": model_names,
        }

    def _build_source_preview(self, hit: RetrievalHit) -> str:
        metadata = hit.metadata or {}
        section = str(metadata.get("section_title") or metadata.get("section") or "").strip()
        section_label = str(metadata.get("section_label") or metadata.get("paragraph_label") or "").strip()
        page = metadata.get("page", metadata.get("start_page"))

        parts = [
            f"source_type={hit.source_type}",
            f"chunk_id={hit.chunk_id}",
            f"rank={hit.rank}",
        ]
        if section_label:
            parts.append(f"section_label={section_label}")
        if section:
            parts.append(f"section_title={section}")
        if page is not None:
            parts.append(f"page={page}")

        return "; ".join(parts)

    def _effective_abstention_config(self, request: GenerateRequest) -> dict[str, Any]:
        override = request.abstention
        return {
            "min_hits": (
                override.min_hits
                if override and override.min_hits is not None
                else self.runtime_config.min_hits_default
            ),
            "min_best_rrf_score": (
                override.min_best_rrf_score
                if override and override.min_best_rrf_score is not None
                else self.runtime_config.min_best_rrf_score_default
            ),
            "min_dual_backend_hits": (
                override.min_dual_backend_hits
                if override and override.min_dual_backend_hits is not None
                else self.runtime_config.min_dual_backend_hits_default
            ),
            "abstain_on_retrieval_errors": (
                override.abstain_on_retrieval_errors
                if override and override.abstain_on_retrieval_errors is not None
                else self.runtime_config.abstain_on_retrieval_errors_default
            ),
            "abstain_on_backend_degradation": (
                override.abstain_on_backend_degradation
                if override and override.abstain_on_backend_degradation is not None
                else self.runtime_config.abstain_on_backend_degradation_default
            ),
        }

    def _pre_llm_abstention_decision(
        self,
        *,
        hits: list[RetrievalHit],
        retrieval_flags: list[RetrievalQualityFlag],
        abstention_cfg: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if not hits:
            return True, "NO_RETRIEVAL_HITS"

        if abstention_cfg["abstain_on_retrieval_errors"]:
            for flag in retrieval_flags:
                if flag.severity == "error":
                    return True, "RETRIEVAL_ERROR_FLAG"

        if abstention_cfg["abstain_on_backend_degradation"]:
            for flag in retrieval_flags:
                if flag.code in {"BACKEND_UNREACHABLE", "BACKEND_HTTP_ERROR"}:
                    return True, "RETRIEVAL_BACKEND_DEGRADED"

        if len(hits) < int(abstention_cfg["min_hits"]):
            return True, "NOT_ENOUGH_HITS"

        best_rrf = max(hit.rrf_score for hit in hits)
        if best_rrf < float(abstention_cfg["min_best_rrf_score"]):
            return True, "LOW_RELEVANCE_SCORE"

        dual_backend_hits = sum(
            1
            for hit in hits
            if hit.sparse_rank is not None and hit.vector_rank is not None
        )
        if dual_backend_hits < int(abstention_cfg["min_dual_backend_hits"]):
            return True, "INSUFFICIENT_DUAL_BACKEND_EVIDENCE"

        return False, None

    def _format_context_piece(self, hit: RetrievalHit, text: str) -> str:
        return (
            f"[CONTEXT]\n"
            f"{self._build_source_preview(hit)}\n"
            f"text:\n{text.strip()}\n"
        )

    def _select_context_hits(
        self,
        *,
        hits: list[RetrievalHit],
        context_budget_tokens: int,
        generator_flags: list[GenerationQualityFlag],
    ) -> tuple[list[UsedChunk], str, int]:
        selected_chunks: list[UsedChunk] = []
        context_parts: list[str] = []
        used_context_tokens = 0

        for hit in hits:
            original_text = hit.text.strip()
            if not original_text:
                continue

            full_piece = self._format_context_piece(hit, original_text)
            full_piece_tokens = self._token_estimator.count(full_piece)

            if used_context_tokens + full_piece_tokens <= context_budget_tokens:
                selected_text = original_text
                selected_piece = full_piece
                selected_tokens = full_piece_tokens
                truncated = False
            else:
                remaining = context_budget_tokens - used_context_tokens
                if remaining <= 32:
                    break

                # Keep metadata header and truncate only the body for a deterministic budget fit.
                header = (
                    "[CONTEXT]\n"
                    f"{self._build_source_preview(hit)}\n"
                    "text:\n"
                )
                header_tokens = self._token_estimator.count(header)
                body_budget = remaining - header_tokens - 1
                if body_budget < 16:
                    break

                selected_text = self._token_estimator.truncate(original_text, body_budget)
                if not selected_text:
                    break

                selected_piece = self._format_context_piece(hit, selected_text)
                selected_tokens = self._token_estimator.count(selected_piece)
                truncated = True

                if used_context_tokens + selected_tokens > context_budget_tokens:
                    selected_text = self._token_estimator.truncate(selected_text, max(body_budget - 16, 1))
                    selected_piece = self._format_context_piece(hit, selected_text)
                    selected_tokens = self._token_estimator.count(selected_piece)
                    if used_context_tokens + selected_tokens > context_budget_tokens:
                        break

            selected_chunks.append(
                UsedChunk(
                    rank=hit.rank,
                    source_type=hit.source_type,
                    chunk_id=hit.chunk_id,
                    chunk_type=hit.chunk_type,
                    text=selected_text,
                    token_count_original=hit.token_count,
                    token_count_used=selected_tokens,
                    truncated=truncated,
                    metadata=hit.metadata,
                    input_file=hit.input_file,
                    rrf_score=hit.rrf_score,
                    sparse_rank=hit.sparse_rank,
                    sparse_score=hit.sparse_score,
                    vector_rank=hit.vector_rank,
                    vector_score=hit.vector_score,
                    vector_distance=hit.vector_distance,
                )
            )
            context_parts.append(selected_piece)
            used_context_tokens += selected_tokens

        if not selected_chunks:
            generator_flags.append(
                GenerationQualityFlag(
                    code="CONTEXT_EMPTY_AFTER_BUDGETING",
                    message="No retrieval context could be selected within the token budget.",
                    severity="warning",
                )
            )

        return selected_chunks, "\n".join(context_parts).strip(), used_context_tokens

    def _build_prompts(self, *, question: str, context_block: str) -> tuple[str, str]:
        # Consider initializing PromptBuilder in __init__ if prompts_dir is in config
        # For now, using default behavior or we can add it to config later
        from generator.prompts import PromptBuilder
        builder = PromptBuilder() 
        return builder.build_system_prompt(), builder.build_user_prompt(question, context_block)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        query = request.query.strip()
        if not query:
            raise ValueError("query must not be empty")

        document_id = request.document_id or self.runtime_config.document_id

        top_k = request.top_k if request.top_k is not None else self.runtime_config.top_k_default
        if top_k < 1 or top_k > self.runtime_config.top_k_max:
            raise ValueError(f"top_k must be between 1 and {self.runtime_config.top_k_max}")

        model = (request.model or self.runtime_config.ollama_model).strip()
        if not model:
            raise ValueError("model must not be empty")

        temperature = (
            request.temperature
            if request.temperature is not None
            else self.runtime_config.temperature_default
        )

        total_budget_tokens = (
            request.total_budget_tokens
            if request.total_budget_tokens is not None
            else self.runtime_config.total_budget_tokens_default
        )
        if total_budget_tokens > self.runtime_config.total_budget_tokens_max:
            raise ValueError(
                f"total_budget_tokens must be <= {self.runtime_config.total_budget_tokens_max}"
            )

        max_answer_tokens = (
            request.max_answer_tokens
            if request.max_answer_tokens is not None
            else self.runtime_config.max_answer_tokens_default
        )
        if max_answer_tokens > self.runtime_config.max_answer_tokens_max:
            raise ValueError(
                f"max_answer_tokens must be <= {self.runtime_config.max_answer_tokens_max}"
            )

        prompt_overhead_tokens = self.runtime_config.prompt_overhead_tokens
        context_budget_tokens = total_budget_tokens - max_answer_tokens - prompt_overhead_tokens
        if context_budget_tokens < 128:
            raise ValueError(
                "Token budget too small. Increase total_budget_tokens or reduce max_answer_tokens."
            )

        retrieval = self._call_retrieval_search(request=request, document_id=document_id, top_k=top_k)

        generator_flags: list[GenerationQualityFlag] = []
        abstention_cfg = self._effective_abstention_config(request)

        pre_abstain, pre_reason = self._pre_llm_abstention_decision(
            hits=retrieval.hits,
            retrieval_flags=retrieval.quality_flags,
            abstention_cfg=abstention_cfg,
        )

        selected_chunks, context_block, used_context_tokens = self._select_context_hits(
            hits=retrieval.hits,
            context_budget_tokens=context_budget_tokens,
            generator_flags=generator_flags,
        )

        if not selected_chunks:
            pre_abstain = True
            pre_reason = pre_reason or "NO_CONTEXT_CHUNKS"

        answer = self.runtime_config.abstention_answer_text
        abstained = pre_abstain
        abstention_reason = pre_reason

        if not pre_abstain:
            system_prompt, user_prompt = self._build_prompts(question=query, context_block=context_block)
            llm_output, raw_content, llm_error = self._call_ollama_chat(
                model=model,
                temperature=temperature,
                total_budget_tokens=total_budget_tokens,
                max_answer_tokens=max_answer_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if llm_error is not None:
                generator_flags.append(
                    GenerationQualityFlag(
                        code="LLM_CALL_FAILED",
                        message="Generator model call failed. Fallback to abstention.",
                        severity="error",
                        context={"error": llm_error, "raw_content": raw_content[:2000]},
                    )
                )
                abstained = True
                abstention_reason = "LLM_CALL_FAILED"
            elif llm_output is None:
                generator_flags.append(
                    GenerationQualityFlag(
                        code="LLM_OUTPUT_INVALID",
                        message="Generator model output could not be parsed. Fallback to abstention.",
                        severity="error",
                    )
                )
                abstained = True
                abstention_reason = "LLM_OUTPUT_INVALID"
            else:
                llm_answer = llm_output.answer.strip()
                llm_abstain = bool(llm_output.should_abstain)

                if llm_output.confidence == "low" and self.runtime_config.abstain_on_low_confidence_default:
                    llm_abstain = True
                    if not abstention_reason:
                        abstention_reason = "LOW_LLM_CONFIDENCE"

                if llm_abstain:
                    abstained = True
                    answer = self.runtime_config.abstention_answer_text
                    abstention_reason = abstention_reason or "LLM_ABSTAINED"
                else:
                    abstained = False
                    answer = llm_answer or self.runtime_config.abstention_answer_text
                    abstention_reason = None

                generator_flags.append(
                    GenerationQualityFlag(
                        code="LLM_REASONING",
                        message=llm_output.reasoning.strip() or "LLM returned structured output.",
                        severity="info",
                        context={"confidence": llm_output.confidence},
                    )
                )

        if abstained:
            answer = self.runtime_config.abstention_answer_text

        return GenerateResponse(
            document_id=document_id,
            query=query,
            answer=answer,
            abstained=abstained,
            abstention_reason=abstention_reason,
            model=model,
            temperature=float(temperature),
            top_k=top_k,
            retrieval_hit_count=len(retrieval.hits),
            total_budget_tokens=total_budget_tokens,
            max_answer_tokens=max_answer_tokens,
            prompt_overhead_tokens=prompt_overhead_tokens,
            context_budget_tokens=context_budget_tokens,
            used_context_tokens=used_context_tokens,
            used_chunks=selected_chunks if request.include_used_chunks else [],
            retrieval_quality_flags=retrieval.quality_flags,
            generator_quality_flags=generator_flags,
            generated_at=datetime.now(timezone.utc),
        )

    def generate_from_retrieval(
        self,
        *,
        request: GenerateRequest,
        retrieval: RetrievalSearchResponse,
    ) -> GenerateResponse:
        query = request.query.strip()
        if not query:
            raise ValueError("query must not be empty")

        document_id = request.document_id or self.runtime_config.document_id

        top_k = request.top_k if request.top_k is not None else self.runtime_config.top_k_default
        if top_k < 1 or top_k > self.runtime_config.top_k_max:
            raise ValueError(f"top_k must be between 1 and {self.runtime_config.top_k_max}")

        model = (request.model or self.runtime_config.ollama_model).strip()
        if not model:
            raise ValueError("model must not be empty")

        temperature = (
            request.temperature
            if request.temperature is not None
            else self.runtime_config.temperature_default
        )

        total_budget_tokens = (
            request.total_budget_tokens
            if request.total_budget_tokens is not None
            else self.runtime_config.total_budget_tokens_default
        )
        if total_budget_tokens > self.runtime_config.total_budget_tokens_max:
            raise ValueError(
                f"total_budget_tokens must be <= {self.runtime_config.total_budget_tokens_max}"
            )

        max_answer_tokens = (
            request.max_answer_tokens
            if request.max_answer_tokens is not None
            else self.runtime_config.max_answer_tokens_default
        )
        if max_answer_tokens > self.runtime_config.max_answer_tokens_max:
            raise ValueError(
                f"max_answer_tokens must be <= {self.runtime_config.max_answer_tokens_max}"
            )

        prompt_overhead_tokens = self.runtime_config.prompt_overhead_tokens
        context_budget_tokens = total_budget_tokens - max_answer_tokens - prompt_overhead_tokens
        if context_budget_tokens < 128:
            raise ValueError(
                "Token budget too small. Increase total_budget_tokens or reduce max_answer_tokens."
            )

        if retrieval.document_id != document_id:
            raise ValueError(
                "retrieval.document_id does not match requested document_id."
            )

        generator_flags: list[GenerationQualityFlag] = []
        abstention_cfg = self._effective_abstention_config(request)

        pre_abstain, pre_reason = self._pre_llm_abstention_decision(
            hits=retrieval.hits,
            retrieval_flags=retrieval.quality_flags,
            abstention_cfg=abstention_cfg,
        )

        selected_chunks, context_block, used_context_tokens = self._select_context_hits(
            hits=retrieval.hits,
            context_budget_tokens=context_budget_tokens,
            generator_flags=generator_flags,
        )

        if not selected_chunks:
            pre_abstain = True
            pre_reason = pre_reason or "NO_CONTEXT_CHUNKS"

        answer = self.runtime_config.abstention_answer_text
        abstained = pre_abstain
        abstention_reason = pre_reason

        if not pre_abstain:
            system_prompt, user_prompt = self._build_prompts(question=query, context_block=context_block)
            llm_output, raw_content, llm_error = self._call_ollama_chat(
                model=model,
                temperature=temperature,
                total_budget_tokens=total_budget_tokens,
                max_answer_tokens=max_answer_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if llm_error is not None:
                generator_flags.append(
                    GenerationQualityFlag(
                        code="LLM_CALL_FAILED",
                        message="Generator model call failed. Fallback to abstention.",
                        severity="error",
                        context={"error": llm_error, "raw_content": raw_content[:2000]},
                    )
                )
                abstained = True
                abstention_reason = "LLM_CALL_FAILED"
            elif llm_output is None:
                generator_flags.append(
                    GenerationQualityFlag(
                        code="LLM_OUTPUT_INVALID",
                        message="Generator model output could not be parsed. Fallback to abstention.",
                        severity="error",
                    )
                )
                abstained = True
                abstention_reason = "LLM_OUTPUT_INVALID"
            else:
                llm_answer = llm_output.answer.strip()
                llm_abstain = bool(llm_output.should_abstain)

                if llm_output.confidence == "low" and self.runtime_config.abstain_on_low_confidence_default:
                    llm_abstain = True
                    if not abstention_reason:
                        abstention_reason = "LOW_LLM_CONFIDENCE"

                if llm_abstain:
                    abstained = True
                    answer = self.runtime_config.abstention_answer_text
                    abstention_reason = abstention_reason or "LLM_ABSTAINED"
                else:
                    abstained = False
                    answer = llm_answer or self.runtime_config.abstention_answer_text
                    abstention_reason = None

                generator_flags.append(
                    GenerationQualityFlag(
                        code="LLM_REASONING",
                        message=llm_output.reasoning.strip() or "LLM returned structured output.",
                        severity="info",
                        context={"confidence": llm_output.confidence},
                    )
                )

        if abstained:
            answer = self.runtime_config.abstention_answer_text

        return GenerateResponse(
            document_id=document_id,
            query=query,
            answer=answer,
            abstained=abstained,
            abstention_reason=abstention_reason,
            model=model,
            temperature=float(temperature),
            top_k=top_k,
            retrieval_hit_count=len(retrieval.hits),
            total_budget_tokens=total_budget_tokens,
            max_answer_tokens=max_answer_tokens,
            prompt_overhead_tokens=prompt_overhead_tokens,
            context_budget_tokens=context_budget_tokens,
            used_context_tokens=used_context_tokens,
            used_chunks=selected_chunks if request.include_used_chunks else [],
            retrieval_quality_flags=retrieval.quality_flags,
            generator_quality_flags=generator_flags,
            generated_at=datetime.now(timezone.utc),
        )

    def readiness(self) -> dict[str, Any]:
        retrieval_ok, retrieval_payload = self._call_retrieval_ready()
        ollama_ok, ollama_payload = self._call_ollama_ready()

        status = "ready" if retrieval_ok and ollama_ok else "degraded"
        return {
            "status": status,
            "dependencies": {
                "retrieval": retrieval_payload,
                "ollama": ollama_payload,
            },
            "defaults": {
                "document_id": self.runtime_config.document_id,
                "model": self.runtime_config.ollama_model,
                "top_k": self.runtime_config.top_k_default,
                "retrieval_rebuild_if_stale": self.runtime_config.retrieval_rebuild_if_stale_default,
                "total_budget_tokens": self.runtime_config.total_budget_tokens_default,
                "max_answer_tokens": self.runtime_config.max_answer_tokens_default,
            },
        }
