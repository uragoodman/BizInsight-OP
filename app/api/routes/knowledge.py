from __future__ import annotations

"""知识库运维接口。

本文件提供 RAG 知识库相关的后端接口，负责让前端或运维脚本通过 HTTP
查看当前项目知识文档，并触发 Qdrant 知识库增量更新。

结构与意义：
- `GET /knowledge/documents`：读取 `docs/knowledge` 下的 Markdown 知识文档，
  返回文档元数据和可选正文，方便确认当前 RAG 知识源。
- `POST /knowledge/index`：调用 RAG 索引构建函数，把变更后的知识文档写入
  Qdrant。实际的切分、hash、manifest 和入库逻辑仍由 `app.rag.indexing` 负责。

使用场景：
- 本地开发时，在更新 `docs/knowledge` 后通过接口触发知识库更新。
- 后续管理后台可以调用这些接口，形成“编辑知识文档 -> 更新向量库”的闭环。
- 面试展示时，可以说明 RAG 不只依赖手动脚本，也具备后端运维入口。
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import require_tenant_context
from app.iam.service import require_permission
from app.rag import indexing
from app.rag.config import get_rag_settings
from app.tenancy import TenantContext


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/documents")
def list_knowledge_documents(
    include_text: bool = Query(default=False),
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """读取当前 RAG 知识文档清单。

    参数：
    - `include_text`：是否返回 Markdown 正文。默认 `False`，避免接口响应过大；
      调试知识内容时可以显式传 `?include_text=true`。

    返回值：
    - `dict[str, object]`：包含知识库根目录、索引版本、文档数量和文档元数据列表。

    具体操作：
    1. 读取 RAG 配置，确认当前知识库根目录和索引版本。
    2. 调用 `load_knowledge_documents` 加载 Markdown 文档。
    3. 将文档转换为接口可读的字典，默认只返回元数据。
    """

    try:
        require_permission(context, "read")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    rag_settings = get_rag_settings()
    documents = indexing.load_knowledge_documents()
    items = []
    for document in documents:
        item = {
            "doc_id": document.doc_id,
            "title": document.title,
            "source_path": document.source_path,
            "doc_type": document.doc_type,
            "metric_name": document.metric_name,
            "tags": document.tags,
            "document_hash": document.document_hash,
            "index_version": document.index_version,
            "text_length": len(document.text),
        }
        if include_text:
            item["text"] = document.text
        items.append(item)

    return {
        "knowledge_root": rag_settings.knowledge_root,
        "collection": rag_settings.collection_name,
        "index_version": rag_settings.index_version,
        "documents": items,
        "total": len(items),
    }


@router.post("/index")
def update_knowledge_index(
    collection: str = Query(..., min_length=1),
    dry_run: bool = Query(default=False),
    context: TenantContext = Depends(require_tenant_context),
) -> dict[str, object]:
    """触发 RAG 知识库增量索引。

    参数：
    - `collection`：显式指定目标 collection，避免运维接口误写当前已验证 collection。
    - `dry_run`：只检查目标状态，不执行写入或删除。
    当前版本固定读取配置中的 `RAG_KNOWLEDGE_ROOT`，避免调用方传入任意文件路径。

    返回值：
    - `dict[str, object]`：`build_qdrant_index` 的执行结果，包括状态、collection、
      写入数量、变更文档数量和索引版本等字段。

    具体操作：
    1. 调用 `app.rag.indexing.build_qdrant_index(collection_name=collection, ...)`。
    2. 由索引层自行完成 Markdown 读取、chunk 切分、document_hash 对比、
       Qdrant 写入和 manifest 更新。
    3. 将索引层结果原样返回给调用方，方便前端或运维脚本展示执行状态。
    """

    try:
        require_permission(context, "manage_knowledge")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return indexing.build_qdrant_index(collection_name=collection, dry_run=dry_run)
