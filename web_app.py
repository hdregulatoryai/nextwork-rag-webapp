# Initiate by running this command: python -m uvicorn app.web_app:app --reload

# Ask a question about the knowledge base:
# http://127.0.0.1:8000/bedrock/query?text=who%20is%20madonna

# Ask a general question:
# http://127.0.0.1:8000/bedrock/invoke?text=who%20is%20madonna

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
import json
import re
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from botocore.exceptions import BotoCoreError, ClientError

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Env ----------
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv("MODEL_ID")  # For /bedrock/invoke
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ARN = os.getenv("MODEL_ARN")

if not AWS_REGION:
    raise ValueError("AWS_REGION environment variable is missing.")

# ---------- App ----------
app = FastAPI(title="RAG Web App", version="1.1.0")

# --- CORS (place BEFORE any @app.get/@app.post routes) ---
import os
from fastapi.middleware.cors import CORSMiddleware

origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
origins = [o.strip() for o in origins_env.split(",") if o.strip()]  # explicit allowlist
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX")  # e.g., r"^https://.*--your-site\.netlify\.app$"

cors_kwargs = dict(
    allow_origins=origins,                # explicit list from env
    allow_credentials=allow_credentials,  # keep FALSE unless you truly need cookies
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],                  # ← important: avoid surprise preflight failures
    expose_headers=["*"],                 # optional; fine to keep
    max_age=600,                          # cache preflight for 10 min
)

# If you want to allow a preview-domain pattern in addition to the explicit list:
if origin_regex:
    cors_kwargs["allow_origin_regex"] = origin_regex

app.add_middleware(CORSMiddleware, **cors_kwargs)

# (Optional but helpful) Fast path for preflight on RAG endpoint
from fastapi import Response
@app.options("/bedrock/query")
def options_query():
    return Response(status_code=204)

# (Optional) Simple health probe to separate reachability from app logic
from datetime import datetime
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}
# --- end CORS ---

# --- region hint (lightweight) ---
def _region_hint(q: str) -> str | None:
    ql = q.lower()

    # EU signals (word-boundary aware)
    eu_patterns = [
        r"\beu\b",
        r"\be\.u\.\b",
        r"\beuropean union\b",
        r"\bmdr\b",
        r"\bivdr\b",
        r"\bce[-\s]*mark\b",          # CE mark / CE-mark
        r"\bnotified\s+body\b",
        r"\beudamed\b",
    ]

    # US signals (word-boundary aware)
    us_patterns = [
        r"\bus\b",
        r"\bu\.s\.\b",
        r"\busa\b",
        r"\bunited\s+states\b",
        r"\bfda\b",
        r"\b21\s*cfr\b",
        r"\b510\s*\(k\)\b",           # 510(k)
        r"\bpma\b",
        r"\bhde\b",
        r"\bde\s+novo\b",
    ]

    eu_hits = any(re.search(p, ql) for p in eu_patterns)
    us_hits = any(re.search(p, ql) for p in us_patterns)

    if eu_hits and not us_hits:
        return "EU"
    if us_hits and not eu_hits:
        return "US"
    return None

def _product_hint(q: str) -> str | None:
    ql = q.lower()

    # Edit these lists anytime to tune detection
    MD_SIGNS = [
        "medical device", "medical-device", "510(k)", "pma", "de novo",
        "class i device", "class ii device", "class iii device", "udise",
        "mdd", "mdsap"
    ]
    IVD_SIGNS = [
        "ivd", "in vitro diagnostic", "in-vitro diagnostic", "ivdr",
        "ivd reagent", "performance evaluation", "eqa"
    ]

    md  = any(w in ql for w in MD_SIGNS)
    ivd = any(w in ql for w in IVD_SIGNS)

    if md and not ivd:
        return "MD"
    if ivd and not md:
        return "IVD"
    return None

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- AWS Clients ----------
try:
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
except (BotoCoreError, ClientError) as e:
    logger.error(f"Failed to initialize AWS clients: {e}")
    raise

# ---------- Helpers ----------
def _last_path_segment(uri: Optional[str]) -> str:
    if not uri:
        return ""
    u = uri.rstrip("/")
    return u.split("/")[-1] if "/" in u else u

def _parse_citations(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts:
      - sources: list[str] of filenames (last path segment from S3 URI)
      - attributions: list[dict] with filename, uri, score, snippet (when available)
    Handles missing fields and minor shape variations across Bedrock responses.
    """
    sources: List[str] = []
    attributions: List[Dict[str, Any]] = []

    for cit in resp.get("citations", []) or []:
        for ref in cit.get("retrievedReferences", []) or []:
            loc = ref.get("location", {}) or {}
            s3_loc = loc.get("s3Location", {}) or {}
            uri = s3_loc.get("uri")

            # score may appear in different places
            score = ref.get("score") or (ref.get("metadata", {}) or {}).get("score")

            # snippet/content may vary
            snippet = None
            content = ref.get("content")
            if isinstance(content, dict):
                snippet = content.get("text") or (content.get("document") or {}).get("text")

            filename = _last_path_segment(uri) if uri else None
            if filename:
                sources.append(filename)

            attributions.append({
                "filename": filename,
                "uri": uri,
                "score": score,
                "snippet": snippet
            })

    # de-dupe sources, preserve order
    seen = set()
    deduped_sources: List[str] = []
    for s in sources:
        if s and s not in seen:
            seen.add(s)
            deduped_sources.append(s)

    return {"sources": deduped_sources, "attributions": attributions}

# ---------- Endpoints ----------
@app.get("/healthz")
async def health():
    return {"ok": True, "region": AWS_REGION}

@app.get("/bedrock/invoke")
async def invoke_model(text: str = Query(..., description="Input text for the model")):
    """
    Direct model invocation (no knowledge base). Returns a stable JSON payload:
      { "response": str, "sources": [], "attributions": [] }
    """
    if not MODEL_ID:
        raise HTTPException(status_code=500, detail="MODEL_ID is not configured.")

    try:
        # Simple prompt format for Llama 3 on Bedrock
        formatted_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{text}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

        request_payload = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
        }

        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_payload),
        )

        body = json.loads(response["body"].read().decode("utf-8"))
        generated_text = body.get("generation", "")

        if not generated_text:
            logger.error("Model did not return any content.")
            raise HTTPException(status_code=500, detail="Model did not return any content.")

        # Keep the same shape as KB endpoint for frontend simplicity
        return {
            "response": generated_text,
            "sources": [],
            "attributions": []
        }

    except ClientError as e:
        logger.exception("AWS ClientError during /bedrock/invoke")
        raise HTTPException(status_code=500, detail=f"AWS Client error: {str(e)}")
    except BotoCoreError as e:
        logger.exception("AWS BotoCoreError during /bedrock/invoke")
        raise HTTPException(status_code=500, detail=f"AWS BotoCore error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error during /bedrock/invoke")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/bedrock/query")
async def query_with_knowledge_base(
    text: str = Query(..., description="Input text for the model"),
    sessionId: Optional[str] = Query(None, description="Reuse this to keep context"),
    newConversation: bool = Query(False, description="Force a fresh KB session")
):
    """
    Retrieval-augmented endpoint using a Bedrock Knowledge Base.

    Returns:
      {
        "response": str,
        "sessionId": str,                  # <-- always included now
        "sources": [ "file1.pdf", ... ],
        "attributions": [
           { "filename": "...", "uri": "...", "score": 0.87, "snippet": "..." },
           ...
        ]
      }
    """
    if not KNOWLEDGE_BASE_ID or not MODEL_ARN:
        raise HTTPException(status_code=500, detail="Knowledge base configuration is missing.")

    try:
        # Only pass sessionId if present and not forcing a new conversation
        effective_sid = None if newConversation else sessionId

        chosen_region  = _region_hint(text)
        chosen_product = _product_hint(text)

        prefix_parts = []

        # REGION: make region primary (optional: repeat once for extra weight)
        if chosen_region:
            prefix_parts.append(f"[REGION:{chosen_region}]")
            prefix_parts.append(f"[REGION:{chosen_region}]")  # repeat = slight extra weight

        # PRODUCT: boost target product AND COMMON
        if chosen_product == "MD":
            prefix_parts.append("[PRODUCT:MD]")
            prefix_parts.append("[PRODUCT:COMMON]")  # <- boost common with MD
        elif chosen_product == "IVD":
            prefix_parts.append("[PRODUCT:IVD]")
            prefix_parts.append("[PRODUCT:COMMON]")  # <- boost common with IVD
        # if ambiguous or none: no product hints (unchanged behavior)

        if prefix_parts:
            text = " ".join(prefix_parts) + " " + text

        kwargs = {
            "input": {"text": text},
            "retrieveAndGenerateConfiguration": {
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                },
                "type": "KNOWLEDGE_BASE",
            },
        }
        if effective_sid:
            kwargs["sessionId"] = effective_sid

        resp = bedrock_agent_client.retrieve_and_generate(**kwargs)

        output_text = (resp.get("output") or {}).get("text", "")
        # Bedrock returns the authoritative sessionId in the response on non-streaming calls
        returned_sid = resp.get("sessionId") or effective_sid or ""

        parsed = _parse_citations(resp)

        return {
            "response": output_text,
            "sessionId": returned_sid,             # <-- NEW
            "sources": parsed["sources"],
            "attributions": parsed["attributions"],
        }

    except ClientError as e:
        logger.exception("AWS ClientError during /bedrock/query")
        raise HTTPException(status_code=500, detail="AWS Client error occurred.")
    except BotoCoreError as e:
        logger.exception("AWS BotoCoreError during /bedrock/query")
        raise HTTPException(status_code=500, detail="AWS BotoCore error occurred.")
    except Exception as e:
        logger.exception("Unexpected error during /bedrock/query")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# === BEGIN: POST variant for /bedrock/query ===
class KBQueryBody(BaseModel):
    text: str
    sessionId: Optional[str] = None
    newConversation: Optional[bool] = False

@app.post("/bedrock/query")
async def query_with_knowledge_base_post(body: KBQueryBody):
    # Reuse the GET logic so there’s only one source of truth
    return await query_with_knowledge_base(
        text=body.text,
        sessionId=body.sessionId,
        newConversation=bool(body.newConversation),
    )
# === END: POST variant for /bedrock/query ===

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)