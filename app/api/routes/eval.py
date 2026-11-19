from fastapi import APIRouter

from app.harness.evaluator import run_evaluation

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run")
def eval_run() -> dict:
    return run_evaluation()
