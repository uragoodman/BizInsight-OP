from app.auth.context import AuthorizationContext
from app.rag.retriever import _filter_chunks
from app.rag.schemas import KnowledgeChunk
from app.tenancy import SellerGrant


def _authz(role="analyst", classifications=("public", "internal")):
    return AuthorizationContext(
        tenant_id="tenant-a", user_id="user-a", role=role, scope_type="seller",
        permission_version="p", policy_set_version="ps", authz_ref="a",
        authz_digest="d", allowed_seller_ids=("seller-a",), seller_grants=(SellerGrant("seller-a", None, None),),
        allowed_metric_ids=("gmv",), allowed_dimensions=("seller",), allowed_actions=("read",),
        allowed_tools=("query_gmv",), allowed_classifications=classifications,
    )


def test_rag_filter_applies_classification_and_role_before_retrieval():
    chunks = [
        KnowledgeChunk(chunk_id="public", doc_id="1", title="p", text="p", source_path="p", classification="public"),
        KnowledgeChunk(chunk_id="conf", doc_id="2", title="c", text="c", source_path="c", classification="confidential"),
        KnowledgeChunk(chunk_id="role", doc_id="3", title="r", text="r", source_path="r", allowed_roles=["admin"]),
    ]
    result = _filter_chunks(chunks, None, None, authorization_context=_authz())
    assert [chunk.chunk_id for chunk in result] == ["public"]
