"""FastAPI server for the MIA24 interactive pipeline demo.

Run:  python demo/app.py
Then: open http://localhost:8000
"""

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root so `from src.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo.pipeline.runner import DemoRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────

DEMO_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = DEMO_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

runner: Optional[DemoRunner] = None

# Session storage (image_id -> intermediate pipeline state)
sessions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    logger.info("Initializing pipeline runner (loading FNDDS data)...")
    runner = DemoRunner()
    logger.info("Server ready")
    yield


app = FastAPI(title="MIA24 Pipeline Demo", lifespan=lifespan)


# Serve frontend
app.mount(
    "/static",
    StaticFiles(directory=DEMO_DIR / "static"),
    name="static",
)
app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads",
)


@app.get("/")
def index():
    return FileResponse(DEMO_DIR / "static" / "index.html")


# ── Request / Response models ─────────────────────────────────

class StepRequest(BaseModel):
    image_id: str


class AnswersRequest(BaseModel):
    image_id: str
    answers: List[str]


class DescribeResponse(BaseModel):
    description: str


class ClarifyResponse(BaseModel):
    questions: List[str]
    suggested_answers: List[List[str]]


class ExpandResponse(BaseModel):
    expanded_description: str
    queries: List[str]


class RetrieveResponse(BaseModel):
    groups: List[Dict[str, Any]]
    total_count: int


class SelectResponse(BaseModel):
    selected: List[Dict[str, str]]


class WeightResponse(BaseModel):
    items: List[Dict[str, Any]]


class NutritionResponse(BaseModel):
    nutrition: Dict[str, float]


# ── Helpers ───────────────────────────────────────────────────

def _get_session(image_id: str) -> Dict[str, Any]:
    session = sessions.get(image_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _require_runner() -> DemoRunner:
    if runner is None:
        raise HTTPException(status_code=503, detail="Runner not ready")
    return runner


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a food image and start a session."""
    ext = Path(file.filename or "img.jpg").suffix or ".jpg"
    image_id = uuid.uuid4().hex[:8]
    filename = f"{image_id}{ext}"
    filepath = UPLOAD_DIR / filename

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    sessions[image_id] = {"image_path": str(filepath)}
    logger.info("Uploaded image %s -> %s", image_id, filepath.name)

    return {"image_id": image_id, "image_url": f"/uploads/{filename}"}


@app.post("/api/pipeline/describe", response_model=DescribeResponse)
def step_describe(req: StepRequest):
    """Step 1: Vision model describes the food."""
    r = _require_runner()
    session = _get_session(req.image_id)
    image_path = Path(session["image_path"])

    description = r.describe_food(image_path)
    session["description"] = description

    return {"description": description}


@app.post("/api/pipeline/clarify", response_model=ClarifyResponse)
def step_clarify(req: StepRequest):
    """Step 2: Generate clarification questions with suggested answers."""
    r = _require_runner()
    session = _get_session(req.image_id)
    description = session.get("description", "")
    if not description:
        raise HTTPException(400, "Run describe step first")

    questions = r.generate_clarification(description)
    session["questions"] = questions

    image_path = Path(session["image_path"])
    suggested = r.suggest_answers(description, questions, image_path)

    return {"questions": questions, "suggested_answers": suggested}


@app.post("/api/pipeline/expand", response_model=ExpandResponse)
def step_expand(req: AnswersRequest):
    """Step 4: Expand query using user answers."""
    r = _require_runner()
    session = _get_session(req.image_id)
    description = session.get("description", "")
    questions = session.get("questions", [])

    if not questions:
        raise HTTPException(400, "Run clarify step first")

    session["answers"] = req.answers

    expanded_desc, queries = r.expand_query(
        description, questions, req.answers
    )
    session["expanded_description"] = expanded_desc
    session["queries"] = queries

    return {"expanded_description": expanded_desc, "queries": queries}


@app.post("/api/pipeline/retrieve", response_model=RetrieveResponse)
def step_retrieve(req: StepRequest):
    """Step 5: Search ChromaDB per ingredient query."""
    r = _require_runner()
    session = _get_session(req.image_id)
    queries = session.get("queries", [])
    expanded_desc = session.get("expanded_description", "")

    if not queries:
        raise HTTPException(400, "Run expand step first")

    groups = r.retrieve(queries, expanded_desc)

    # Flatten for selection step
    all_candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for group in groups:
        for c in group["candidates"]:
            if c["food_code"] not in seen:
                seen.add(c["food_code"])
                all_candidates.append(c)
    session["candidates"] = all_candidates

    total = sum(len(g["candidates"]) for g in groups)
    return {"groups": groups, "total_count": total}


@app.post("/api/pipeline/select", response_model=SelectResponse)
def step_select(req: StepRequest):
    """Step 6: Vision model selects best FNDDS codes."""
    r = _require_runner()
    session = _get_session(req.image_id)
    candidates = session.get("candidates", [])
    image_path = Path(session["image_path"])

    if not candidates:
        raise HTTPException(400, "Run retrieve step first")

    selected = r.select_food_codes(candidates, image_path)
    session["selected"] = selected

    return {"selected": selected}


@app.post("/api/pipeline/weight", response_model=WeightResponse)
def step_weight(req: StepRequest):
    """Step 7: Estimate weight for each selected food code."""
    r = _require_runner()
    session = _get_session(req.image_id)
    selected = session.get("selected", [])
    image_path = Path(session["image_path"])

    if not selected:
        raise HTTPException(400, "Run select step first")

    items = r.estimate_weights(selected, image_path)
    session["food_items"] = items

    return {"items": items}


@app.post("/api/pipeline/nutrition", response_model=NutritionResponse)
def step_nutrition(req: StepRequest):
    """Step 8: Calculate final nutrition values."""
    r = _require_runner()
    session = _get_session(req.image_id)
    food_items = session.get("food_items", [])

    if not food_items:
        raise HTTPException(400, "Run weight step first")

    nutrition = r.calculate_nutrition(food_items)
    session["nutrition"] = nutrition

    return {"nutrition": nutrition}


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
