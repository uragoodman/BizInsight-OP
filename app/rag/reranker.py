"""RAG cross-encoder 重排器（P3.3）：bge-reranker-base，懒加载 + 结构化降级。

- 只负责相关性排序；权限/租户/范围过滤在检索层已先行完成，模型无法突破治理边界。
- 模型懒加载单例；加载失败/推理超时/非法分数等均返回结构化 RerankResult（reason 区分），
  调用方据此降级到规则排序并记录原因，不静默跳过。
- 语料可扩展：重排器按 (query, chunk.text) 打分，不感知文档/切块规模。
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class RerankCandidate:
    """一个待精排的候选 chunk。"""
    chunk_id: str
    text: str


@dataclass
class RerankResult:
    """结构化的重排结果；reason 区分降级原因供观测/评测使用。"""
    scores: list[float] | None
    applied: bool
    degraded: bool
    reason: str | None
    latency_ms: float = 0.0

    def as_metric(self) -> dict[str, Any]:
        return {
            "applied": self.applied, "degraded": self.degraded,
            "reason": self.reason, "latency_ms": round(self.latency_ms, 2),
        }


class Reranker:
    """CrossEncoder 精排器：懒加载单例、推理超时、结构化降级。"""

    _lock = threading.Lock()
    _instance: "Reranker | None" = None
    _model: Any = None
    _model_failed = False

    @classmethod
    def singleton(cls) -> "Reranker":
        """进程级单例：模型常驻，热启动复用。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> Any | None:
        """懒加载模型；加载失败后不再重试（避免每次检索都尝试下载）。"""
        if self._model is not None or self._model_failed:
            return self._model
        with self._lock:
            if self._model is not None or self._model_failed:
                return self._model
            try:
                # 命中缓存后离线可用；首次下载需允许联网（不强制 HF_HUB_OFFLINE）。
                settings = get_settings()
                os.environ.setdefault("HF_HUB_CACHE", str(settings.rag_cache_dir))
                from sentence_transformers import CrossEncoder

                model_name = settings.rag_reranker_model or RERANKER_MODEL
                revision = settings.rag_reranker_model_revision
                # Resolve the pinned Hub snapshot to a local path when it is
                # already prefetched. This remains reliable when another
                # HuggingFace model imported hub constants before HF_HUB_CACHE
                # was set by this lazy loader.
                snapshot_path = None
                if revision and "/" in model_name and not Path(model_name).is_dir():
                    repo_cache = f"models--{model_name.replace('/', '--')}"
                    candidate = Path(settings.rag_cache_dir) / repo_cache / "snapshots" / revision
                    if candidate.is_dir():
                        snapshot_path = candidate.resolve()
                if snapshot_path is not None:
                    model_name = str(snapshot_path)
                    model_kwargs: dict[str, Any] = {}
                else:
                    model_kwargs = {"cache_folder": str(settings.rag_cache_dir)}
                    if revision:
                        model_kwargs["revision"] = revision
                self._model = CrossEncoder(
                    model_name,
                    **model_kwargs,
                )
            except Exception:  # noqa: BLE001 - 加载失败降级
                self._model_failed = True
                self._model = None
        return self._model

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> RerankResult:
        """对 (query, candidate) 对打分；模型不可用/异常/非法分数时结构化降级。

        是否启用由调用方（检索层按 strategy）决定，本方法只负责模型调用与降级。
        """
        if not candidates:
            return RerankResult(None, False, True, "insufficient_candidates")
        model = self._load()
        if model is None:
            return RerankResult(None, False, True, "model_unavailable")
        pairs = [(query, candidate.text) for candidate in candidates]
        started = time.perf_counter()
        timeout = max(0.05, get_settings().rag_reranker_timeout_ms / 1000.0)

        def _predict() -> list[float]:
            return [float(score) for score in model.predict(pairs)]

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_predict)
                scores = future.result(timeout=timeout)
            if len(scores) != len(candidates) or any(score != score for score in scores):  # NaN 检查
                return RerankResult(None, False, True, "invalid_scores", round((time.perf_counter() - started) * 1000, 2))
            return RerankResult(scores, True, False, None, round((time.perf_counter() - started) * 1000, 2))
        except FutureTimeout:
            return RerankResult(None, False, True, "timeout", round((time.perf_counter() - started) * 1000, 2))
        except Exception:  # noqa: BLE001 - 推理异常降级
            return RerankResult(None, False, True, "runtime_error", round((time.perf_counter() - started) * 1000, 2))


def rerank(query: str, candidates: list[RerankCandidate]) -> RerankResult:
    """模块级便捷入口：复用单例。"""
    return Reranker.singleton().rerank(query, candidates)
