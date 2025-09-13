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

import os, uuid, time
from typing import Dict, Any, Tuple, List
from fastapi import Request
from fastapi.responses import JSONResponse

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float | None) -> float | None:
    val = os.getenv(name)
    if val in (None, "", "none", "None"):
        return default
    try:
        return float(val)
    except Exception:
        return default

# Defaults you can tweak via env (optional)
RAG_STRICT_TOP_K = _env_int("RAG_STRICT_TOP_K", 6)      # first pass
RAG_RELAXED_TOP_K = _env_int("RAG_RELAXED_TOP_K", 12)   # fallback pass
RAG_MIN_SCORE = _env_float("RAG_MIN_SCORE", None)       # e.g., 0.3; None = no threshold

def build_filters_from_prefs(prefs: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Turn your 'bias/preferences' into Bedrock KB filters.
    Return None if you want no filters.
    Example assumes simple OR over tags/region.
    """
    if not prefs:
        return None
    tags = prefs.get("tags") or []
    region = prefs.get("region")  # e.g., "EU" or "US"
    clauses = []
    if tags:
        # Any of these tags allowed (OR)
        clauses.append({"any": [{"equals": {"key": "tag", "value": t}} for t in tags]})
    if region:
        clauses.append({"equals": {"key": "region", "value": region}})
    if not clauses:
        return None
    # AND the high-level conditions if both present; OR within tags
    if len(clauses) == 1:
        return clauses[0]
    return {"all": clauses}

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

# ---- Bedrock KB call used by POST /bedrock/query ----
async def call_bedrock_rag(
    message: str,
    session_id: str | None,
    filters: dict | None,
    top_k: int,
    min_score: float | None,   # not all SDKs support threshold here; we omit if None
) -> tuple[str, list[str]]:
    """
    Calls Bedrock retrieve_and_generate using your existing agent-runtime client.
    Returns:
      output_text: str
      hits: list[str]   # filenames (same as GET /bedrock/query 'sources')
    """

    # Base args (same shape you use in GET /bedrock/query)
    kwargs = {
        "input": {"text": message},
        "retrieveAndGenerateConfiguration": {
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN,
                # We'll optionally add 'retrievalConfiguration' below
            },
            "type": "KNOWLEDGE_BASE",
        },
    }
    if session_id:
        kwargs["sessionId"] = session_id

    # Optional retrieval tuning (only include when present)
    # NOTE: Field names below follow current Bedrock KB API; if your account/SDK
    # version differs, you can comment this whole block and it will still work
    # with default KB settings (no filters, default top_k).
    retrieval_conf = {}
    vector_conf = {}

    if isinstance(top_k, int) and top_k > 0:
        # How many results to retrieve
        vector_conf["numberOfResults"] = top_k

    if filters:
        # Metadata filter constructed by build_filters_from_prefs()
        # API expects singular key 'filter' under vectorSearchConfiguration.
        vector_conf["filter"] = filters

    # (Optional) some SDKs allow a confidence threshold; if yours errors on this,
    # just remove the whole "if min_score..." block.
    if (min_score is not None) and isinstance(min_score, (int, float)):
        # Only include if your KB supports a score threshold; otherwise omit.
        # Commented out by default to avoid API shape mismatches:
        # vector_conf["overrideSearchType"] = "HYBRID"   # example; safe to omit
        # vector_conf["minScore"] = float(min_score)     # example; safe to omit
        pass

    if vector_conf:
        retrieval_conf["vectorSearchConfiguration"] = vector_conf

    if retrieval_conf:
        kwargs["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"][
            "retrievalConfiguration"
        ] = retrieval_conf

    # ---- Call Bedrock and parse ----
    resp = bedrock_agent_client.retrieve_and_generate(**kwargs)

    # Text lives under output.text for agent-runtime
    output_text = (resp.get("output") or {}).get("text", "")
    returned_sid = resp.get("sessionId") or session_id or ""
    # Reuse your existing citation parser to get filenames
    parsed = _parse_citations(resp)
    hits = parsed["sources"]  # list[str] filenames

    return output_text, hits, returned_sid

# === BEGIN: POST variant for /bedrock/query ===
class KBQueryBody(BaseModel):
    text: str
    sessionId: Optional[str] = None
    newConversation: Optional[bool] = False

@app.post("/bedrock/query")
async def kb_query(req: Request):
    # ---- read request ----
    payload = await req.json()

    # Accept old/new payloads: "message" (new) and "text" (old)
    raw_msg = payload.get("message")
    if not raw_msg:
        raw_msg = payload.get("text")

    user_msg = (raw_msg or "").strip()

    if not user_msg:
    return JSONResponse(
        {
            "ok": False,
            "reason": "empty_query",
            "response": "Please enter a question.",
            "sessionId": session_id,
            "sources": [],
            "attributions": []
        },
        status_code=200
    )

    # Accept both "useKnowledgeBase" (new) and default to True
    use_kb = payload.get("useKnowledgeBase")
    if use_kb is None:
        use_kb = True

    # Accept "sessionId" as-is (None/null is fine)
    session_id = payload.get("sessionId")

    # optional query params
    qp = req.query_params
    debug   = qp.get("debug") == "1"
    nobias  = qp.get("nobias") == "1"          # force relaxed only
    strict_only = qp.get("strict_only") == "1" # try strict only, no fallback

    # your existing preference object coming from client (if any)
    prefs = payload.get("preferences") or {}   # e.g., {"region":"EU","tags":["IVDR"]}

    # ---- decide strategies ----
    strategies: List[Dict[str, Any]] = []
    if use_kb and not nobias:
        # STRICT FIRST
        strategies.append({
            "name": "strict",
            "filters": build_filters_from_prefs(prefs),
            "top_k": RAG_STRICT_TOP_K,
            "min_score": RAG_MIN_SCORE,      # None means don't pass a threshold
        })
        if not strict_only:
            # RELAXED FALLBACK
            strategies.append({
                "name": "relaxed",
                "filters": None,              # <- IMPORTANT: drop filters
                "top_k": RAG_RELAXED_TOP_K,
                "min_score": None,           # no threshold on fallback
            })
    else:
        # DIRECTLY RELAXED (nobias or use_kb == False but you still want KB search)
        strategies.append({
            "name": "relaxed",
            "filters": None,
            "top_k": RAG_RELAXED_TOP_K,
            "min_score": None,
        })

    debug_info = {"strategies": [], "query": user_msg[:200]}
    last_error = None
    session_id = payload.get("sessionId")  # keep your existing session handling

    # ---- run strategies in order ----
    for strat in strategies:
        t0 = time.time()
        try:
            # === CALL BEDROCK HERE ===
            # Build your Bedrock RetrieveAndGenerate payload using strat["filters"], strat["top_k"], strat["min_score"]
            # Example pseudo-build (replace with your actual client code):
            # bedrock_req = {
            #   "input": user_msg,
            #   "retrieveConfig": {
            #       "knowledgeBaseId": KB_ID,
            #       "numberOfResults": strat["top_k"],
            #       "minScoreConfidence": strat["min_score"],  # only if supported
            #       "filters": strat["filters"],               # only if supported by your SDK/config
            #   },
            #   "sessionId": session_id,
            # }
            #
            # bedrock_resp = bedrock_client.retrieve_and_generate(**bedrock_req)
            # Parse:
            # output_text = bedrock_resp["outputText"]
            # hits = normalize_sources(bedrock_resp)   # list of your citations
            # hits_count = len(hits)

            # ↓↓↓ replace this mock with your real call ↓↓↓
            output_text, hits, returned_sid = await call_bedrock_rag(
                message=user_msg,
                session_id=session_id,
                filters=strat["filters"],
                top_k=strat["top_k"],
                min_score=strat["min_score"],
            )
            hits_count = len(hits or [])
            # ↑↑↑ replace with your real call ↑↑↑

            debug_info["strategies"].append({
                "name": strat["name"],
                "filters": strat["filters"],
                "top_k": strat["top_k"],
                "min_score": strat["min_score"],
                "hits_count": hits_count,
                "elapsed_ms": int((time.time() - t0) * 1000),
            })

            if hits_count > 0 and (output_text or "").strip():
                # success on this strategy
                out = {
                    "ok": True,
                    "response": output_text,
                    "sessionId": returned_sid,
                    "sources": hits,
                    "attributions": [],  # keep as you had
                }
                if debug:
                    out["debug"] = debug_info
                return JSONResponse(out, status_code=200)

            # else: try next strategy
        except Exception as e:
            last_error = str(e)
            debug_info["strategies"].append({
                "name": strat["name"], "error": last_error,
                "elapsed_ms": int((time.time() - t0) * 1000),
            })
            # try next strategy

    # ---- all strategies failed or zero hits ----
    reason = "no_kb_hits" if last_error is None else "bedrock_error"
    out = {
        "ok": False,
        "reason": reason,
        "response": "Sorry, I am unable to assist you with this request.",
        "sessionId": session_id,
        "sources": [],
        "attributions": []
    }
    if debug:
        if last_error:
            debug_info["last_error"] = last_error
        out["debug"] = debug_info
    return JSONResponse(out, status_code=200)

# === END: POST variant for /bedrock/query ===

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)