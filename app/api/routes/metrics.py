from fastapi import APIRouter, Request, Response

from app.observability.tracing import prometheus_payload

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    # The frontend uses /metrics as a client-side route while Prometheus scrapes
    # the same path. Serve the SPA to browser document requests and preserve the
    # metrics exposition format for scrapers (which send */* or text/plain).
    if "text/html" in request.headers.get("accept", "").lower():
        from app.api.routes.ui import index

        return index()
    return Response(prometheus_payload(), media_type="text/plain; version=0.0.4")
