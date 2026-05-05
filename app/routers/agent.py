

from fastapi import APIRouter, Depends, HTTPException, Request

from app.limiter import limiter
from app.schemas.agent import AgentQueryRequest, AgentQueryResponse
from app.services.agent_service import AgentService


router = APIRouter()


def get_agent_service(request: Request) -> AgentService:
    svc: AgentService | None = getattr(request.app.state, "agent_service", None)
    if svc is None or not svc.enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent service unavailable; NVIDIA_API_KEY not configured",
        )
    return svc


@router.post("/query", response_model=AgentQueryResponse)
@limiter.limit("10/minute")
async def query(
    request: Request,
    body: AgentQueryRequest,
    agent: AgentService = Depends(get_agent_service),
) -> AgentQueryResponse:
    try:
        return await agent.query(body.query, body.portfolio_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
