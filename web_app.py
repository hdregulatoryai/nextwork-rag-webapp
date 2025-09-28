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
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from botocore.exceptions import BotoCoreError, ClientError

import re, json
from typing import Dict, Any, List

# Broader small-talk detector
_SMALL_TALK_RE = re.compile(
    r"""^\s*(?:            # start
         hi(?:\s+there)?|
         hello(?:\s+there)?|
         hey(?:a)?|
         howdy|
         yo|
         sup|
         good\s+(?:morning|afternoon|evening)|
         thanks?|thank\s+you|
         test(?:ing)?|
         👋|🙏
    )\b""",
    re.IGNORECASE | re.VERBOSE
)

# Strong regulatory “not small talk” terms
_REGULATORY_HINTS = re.compile(
    r"(ivdr|mdr|21\s*cfr|part\s*820|qsr|iso\s*13485|iso\s*14971|capa|"
    r"risk\s*management|510\([kK]\)|de\s*novo|fda|eudamed|udi|mdcg|qms)",
    re.IGNORECASE
)

def is_small_talk(msg: str) -> bool:
    if not msg:
        return False
    # normalize whitespace
    m = " ".join(msg.strip().split())
    # hard bail if any regulatory hints appear
    if _REGULATORY_HINTS.search(m):
        return False
    # very short and matches small-talk
    if len(m.split()) <= 8 and _SMALL_TALK_RE.search(m):
        return True
    # extremely short non-informative tokens (e.g., "?", ".", "...", "ok")
    if len(m) <= 3 and re.fullmatch(r"[.?!…]+|ok|kk|k", m.lower()):
        return True
    return False

AUTO_BIAS_ENABLED = os.getenv("AUTO_BIAS_ENABLED", "true").lower() == "true"

def infer_preferences_from_text(text: str) -> Dict[str, Any]:
    """
    Infer lightweight preferences from the user's query.
    Returns {} when nothing confidently matches.
    """
    if not text:
        return {}
    t = text.lower()

    prefs: Dict[str, Any] = {}
    tags: List[str] = []

    # --- EU signals ---
    if re.search(r"\bivdr\b|\bin[-\s]?vitro diagnostic regulation\b", t):
        prefs["region"] = "EU"
        tags += ["IVDR"]
    if re.search(r"\bmdr\b|\bmedical device regulation\b", t):
        # prefer setting EU if not already set by IVDR
        prefs.setdefault("region", "EU")
        tags += ["MDR"]
    if re.search(r"\bnotified body\b|\bce[-\s]?mark", t):
        prefs.setdefault("region", "EU")

    # --- US signals ---
    if re.search(r"\b21\s*cfr\b|\bpart\s*820\b|\bqsr\b", t):
        prefs["region"] = "US"
        tags += ["21 CFR 820", "QSR"]
    if re.search(r"\bfda\b|\b510\(\w\)|\bde novo\b|\bpremarket\b", t):
        prefs.setdefault("region", "US")

    # --- Common frameworks/topics ---
    if re.search(r"\biso\s*13485\b", t):
        tags += ["ISO 13485"]
    if re.search(r"\bpost[-\s]?market|\bpostmarket", t):
        tags += ["postmarket"]
    if re.search(r"\bcapa\b", t):
        tags += ["CAPA"]
    if re.search(r"\brisk management|\biso\s*14971\b", t):
        tags += ["Risk Management"]

    if tags:
        # preserve discovery order but unique
        seen = set()
        uniq = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                uniq.append(tag)
        prefs["tags"] = uniq

    return prefs

def merge_preferences(explicit: Dict[str, Any] | None, inferred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit wins; fill any missing fields from inferred.
    """
    explicit = explicit or {}
    merged = dict(inferred)  # start with inferred
    merged.update({k: v for k, v in explicit.items() if v not in (None, "", [], {})})
    return merged

def build_bias_instructions(prefs: Dict[str, Any]) -> str:
    """
    Produce a short, soft instruction for the model.
    Never forces filtering; just a preference hint.
    prefs example: {"region":"EU", "tags":["IVDR","postmarket"]}
    """
    if not prefs:
        return ""
    bits = []
    region = prefs.get("region")
    tags = prefs.get("tags") or []
    if region:
        bits.append(f"Prefer regulatory sources specific to {region}.")
    if tags:
        bits.append("If multiple sources are relevant, prioritize documents related to: " +
                    ", ".join(tags) + ".")
    if not bits:
        return ""
    # Important: keep it as guidance, not a hard constraint.
    return ("Guidance for retrieval and citation preference (do NOT refuse if unavailable): "
            + " ".join(bits))

def rerank_sources_soft(sources: List[str], prefs: Dict[str, Any]) -> List[str]:
    """
    Soft re-ranking: move preferred items earlier, but never drop anything.
    Assumes each source is a filename or title string; adjust if yours is a dict.
    """
    if not sources or not prefs:
        return sources
    region = (prefs.get("region") or "").lower()
    tags = [t.lower() for t in (prefs.get("tags") or [])]
    def score(s: str) -> int:
        s_l = s.lower()
        sc = 0
        if region and region in s_l:
            sc += 2
        for t in tags:
            if t and t in s_l:
                sc += 1
        return -sc  # Python sorts ascending; negative for higher-is-earlier
    return sorted(sources, key=score)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Env ----------
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MODEL_ID = os.getenv("MODEL_ID")  # For /bedrock/invoke
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ARN = os.getenv("MODEL_ARN")
KB_VERSION = os.getenv("KB_VERSION", "1.0") # Add this new variable
# Define the specific error message to catch for stale session IDs
STALE_SESSION_ERROR_MSG = "Session not found" # Bedrock's typical message for a bad session in ValidationException
# -----------
if not AWS_REGION:
    raise ValueError("AWS_REGION environment variable is missing.")

# ---------- App ----------
app = FastAPI(title="RAG Web App", version="1.1.0")

# CORS: restrict to explicit frontend origins from env
origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
origins = [o.strip() for o in origins_env.split(",") if o.strip()]  # clean + ignore empties

# If nothing provided, you can choose to fail closed or set a safe default list.
# For now, default CLOSED (no cross-origin browser access) to avoid accidental "*" in prod.
if not origins:
    origins = []  # empty list = no cross-origin browser reads allowed

allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
# Optional: support a regex for preview URLs (e.g., Netlify deploy previews)
origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX")  # e.g., r"^https://.*--your-site\.netlify\.app$"

cors_kwargs = dict(
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=allow_credentials,
)
if origin_regex:
    cors_kwargs["allow_origin_regex"] = origin_regex

app.add_middleware(CORSMiddleware, **cors_kwargs)

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

# --- NEW HELPER FUNCTION ---
def _call_retrieve_and_generate(
    text_for_model: str,
    sessionId: Optional[str],
    kb_id: str,
    model_arn: str
) -> Dict[str, Any]:
    """
    Internal function to execute the bedrock retrieve_and_generate call.
    Returns the full Bedrock response dictionary.
    """
    kwargs = {
        "input": {"text": text_for_model},
        "retrieveAndGenerateConfiguration": {
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN,
                
                # --- RETRIEVAL FIX (Confirmed) ---
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        # 1. Initial Retrieval Pool Size (N=30)
                        "numberOfResults": 30, 
                        
                        # 2. Hybrid Search Fix
                        "overrideSearchType": "HYBRID",
                        
                        # 3. Corrected Reranking Block Structure (K=15)
                        "rerankingConfiguration": {
                            # REQUIRED: Specifies the overall type of the ranker
                            "type": "BEDROCK_QUERY_RANKER", 
                            
                            # REQUIRED: Nest the specific settings inside this dictionary
                            "bedrockRerankingConfiguration": {
                                # REQUIRED: Contains configuration for the reranker model (must be present)
                                "modelConfiguration": {}, 
                                
                                # CORRECTED NAME: Re-rank the top 30 and pass the best 15 chunks to the LLM (K=15)
                                "numberOfRerankedResults": 15 
                            }
                        }
                    }
                },
                # --- END OF CONFIGURATION ---
                
            },
            "type": "KNOWLEDGE_BASE",
        },
    }
    if sessionId:
        kwargs["sessionId"] = sessionId

    # Call Bedrock KB
    return bedrock_agent_client.retrieve_and_generate(**kwargs)
# --- END NEW HELPER FUNCTION ---

# ---------- Endpoints ----------
@app.get("/healthz")
async def health():
    return {"ok": True, "region": AWS_REGION}

# --- NEW ENDPOINT ---
@app.get("/kb/info")
async def get_knowledge_base_info():
    """
    Returns the current Knowledge Base ID and a version token for client-side proactive checks.
    """
    if not KNOWLEDGE_BASE_ID:
        # Return an error or a known default if KB is not configured
        raise HTTPException(status_code=503, detail="Knowledge base is not deployed or configured.")
        
    return {
        "kbId": KNOWLEDGE_BASE_ID,
        "kbVersion": KB_VERSION
    }
# --- END NEW ENDPOINT ---

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
    text: str = Query(...),
    sessionId: Optional[str] = Query(None),
    newConversation: bool = Query(False),
    preferences: Optional[str] = Query(None)  # optional JSON string
):
    # parse prefs (POST passes dict via wrapper → here as JSON string)
    prefs: Dict[str, Any] = {}
    if preferences and isinstance(preferences, str):
        try:
            prefs = json.loads(preferences)
        except Exception:
            prefs = {}

    # Auto-infer preferences from text
    inferred = infer_preferences_from_text(text)
    # FIX: Define merged_prefs to ensure preferences are handled correctly and reranking works.
    merged_prefs = merge_preferences(prefs, inferred) if AUTO_BIAS_ENABLED else (prefs or None)

    """
    Retrieval-augmented endpoint using a Bedrock Knowledge Base.
    ...
    """
    if not KNOWLEDGE_BASE_ID or not MODEL_ARN:
        raise HTTPException(status_code=500, detail="Knowledge base configuration is missing.")

    # 1. Compute effective session id
    effective_sid = None if newConversation else (sessionId or None)

    # 2. Short-circuit tiny greetings on first message to avoid KB cold edge
    if is_small_talk(text):
        return {
            "response": "Hi! I’m ready to help. What is your question?",
            "sessionId": effective_sid or "",
            "sources": [],
            "attributions": []
        }

    # 3. Build soft bias + input text
    bias = build_bias_instructions(merged_prefs)
    text_for_model = f"{bias}\n\nUser question: {text}" if bias else text

    # 4. Attempt retrieve and generate with retry logic
    session_reset_flag = False
    bedrock_resp = None

    try:
        # --- ATTEMPT 1: Use the provided session ID ---
        bedrock_resp = _call_retrieve_and_generate(
            text_for_model=text_for_model,
            sessionId=effective_sid,
            kb_id=KNOWLEDGE_BASE_ID,
            model_arn=MODEL_ARN
        )

    except ClientError as e:
        error_message = str(e)
        # Check for the specific Bedrock session error (ValidationException)
        if 'ValidationException' in error_message and STALE_SESSION_ERROR_MSG in error_message:
            logger.warning(f"Stale session detected ({effective_sid}). Retrying without session ID.")
            session_reset_flag = True # Set flag for response
            
            try:
                # --- ATTEMPT 2: Retry without any session ID (Bedrock creates a new one) ---
                bedrock_resp = _call_retrieve_and_generate(
                    text_for_model=text_for_model,
                    sessionId=None, # Crucial: retry with no session ID
                    kb_id=KNOWLEDGE_BASE_ID,
                    model_arn=MODEL_ARN
                )
            except (ClientError, BotoCoreError) as retry_e:
                 logger.error(f"AWS error on retry: {retry_e}")
                 raise HTTPException(status_code=500, detail="AWS error during session reset and retry.")
        else:
            # Re-raise all other ClientErrors
            logger.exception("AWS ClientError during /bedrock/query (non-session error)")
            raise HTTPException(status_code=500, detail=f"AWS Client error occurred: {error_message}")
            
    except BotoCoreError:
        logger.exception("AWS BotoCoreError during /bedrock/query")
        raise HTTPException(status_code=500, detail="AWS BotoCore error occurred.")
    except Exception as e:
        logger.exception("Unexpected error during /bedrock/query")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


    # 5. Process and return response
    if not bedrock_resp:
        raise HTTPException(status_code=500, detail="Failed to get a response from Bedrock after all attempts.")

    # Extract model text + authoritative session id
    output_text = (bedrock_resp.get("output") or {}).get("text", "")
    returned_sid = bedrock_resp.get("sessionId") or ""

    # Parse + soft re-rank sources (non-destructive)
    parsed = _parse_citations(bedrock_resp)
    sources = rerank_sources_soft(parsed["sources"], merged_prefs) # FIX: Use merged_prefs

    response_payload = {
        "response": output_text,
        "sessionId": returned_sid,
        "sources": sources,
        "attributions": parsed["attributions"],
    }
    
    # 6. Add the session_reset flag if a retry occurred
    if session_reset_flag:
        response_payload["session_reset"] = True
    
    return response_payload

# === BEGIN: POST variant for /bedrock/query ===
class KBQueryBody(BaseModel):
    text: str
    sessionId: Optional[str] = None
    newConversation: Optional[bool] = False
    preferences: Optional[Dict[str, Any]] = None   # NEW

# web_app.py (REPLACE the existing @app.post("/bedrock/query") function, around line 324)

@app.post("/bedrock/query")
async def query_with_knowledge_base_post(body: KBQueryBody):
    # Normalize text early
    user_text = (body.text or "").strip()

    if not KNOWLEDGE_BASE_ID or not MODEL_ARN:
        raise HTTPException(status_code=500, detail="Knowledge base configuration is missing.")

    # 1. Compute effective session id
    effective_sid = body.sessionId or None # Use None if empty string

    # 2. Short-circuit tiny greetings
    if is_small_talk(user_text):
        return {
            "response": "Hi! I’m ready to help. What is your question?",
            "sessionId": effective_sid,
            "sources": [],
            "attributions": []
        }

    # 3. Handle preferences/bias
    inferred = infer_preferences_from_text(user_text)
    merged_prefs = merge_preferences(body.preferences, inferred) if AUTO_BIAS_ENABLED else (body.preferences or None)
    
    bias = build_bias_instructions(merged_prefs)
    text_for_model = f"{bias}\n\nUser question: {user_text}" if bias else user_text

    # 4. Attempt retrieve and generate with retry logic
    session_reset_flag = False
    bedrock_resp = None

    try:
        # --- ATTEMPT 1: Use the provided session ID ---
        bedrock_resp = _call_retrieve_and_generate(
            text_for_model=text_for_model,
            sessionId=effective_sid,
            kb_id=KNOWLEDGE_BASE_ID,
            model_arn=MODEL_ARN
        )

    except ClientError as e:
        error_message = str(e)
        # Check for the specific Bedrock session error (ValidationException)
        if 'ValidationException' in error_message and STALE_SESSION_ERROR_MSG in error_message:
            logger.warning(f"Stale session detected ({effective_sid}). Retrying without session ID.")
            session_reset_flag = True # Set flag for response
            
            try:
                # --- ATTEMPT 2: Retry without any session ID (Bedrock creates a new one) ---
                bedrock_resp = _call_retrieve_and_generate(
                    text_for_model=text_for_model,
                    sessionId=None, # Crucial: retry with no session ID
                    kb_id=KNOWLEDGE_BASE_ID,
                    model_arn=MODEL_ARN
                )
            except (ClientError, BotoCoreError) as retry_e:
                 logger.error(f"AWS error on retry: {retry_e}")
                 raise HTTPException(status_code=500, detail="AWS error during session reset and retry.")
        else:
            # Re-raise all other ClientErrors
            logger.exception("AWS ClientError during /bedrock/query (non-session error)")
            raise HTTPException(status_code=500, detail=f"AWS Client error occurred: {error_message}")
            
    except BotoCoreError:
        logger.exception("AWS BotoCoreError during /bedrock/query")
        raise HTTPException(status_code=500, detail="AWS BotoCore error occurred.")
    except Exception as e:
        logger.exception("Unexpected error during /bedrock/query")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


    # 5. Process and return response
    if not bedrock_resp:
        raise HTTPException(status_code=500, detail="Failed to get a response from Bedrock after all attempts.")

    # Extract model text + authoritative session id
    output_text = (bedrock_resp.get("output") or {}).get("text", "")
    returned_sid = bedrock_resp.get("sessionId") or ""

    # Parse + soft re-rank sources (non-destructive)
    parsed = _parse_citations(bedrock_resp)
    sources = rerank_sources_soft(parsed["sources"], merged_prefs)

    response_payload = {
        "response": output_text,
        "sessionId": returned_sid,
        "sources": sources,
        "attributions": parsed["attributions"],
    }
    
    # 6. Add the session_reset flag if a retry occurred
    if session_reset_flag:
        response_payload["session_reset"] = True
    
    return response_payload
# === END: POST variant for /bedrock/query ===

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)