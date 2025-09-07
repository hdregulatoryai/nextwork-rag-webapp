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
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import logging
import time
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal
from botocore.exceptions import BotoCoreError, ClientError

class MsgPart(BaseModel):
    text: str

class Msg(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: List[MsgPart]

class ConverseBody(BaseModel):
    messages: List[Msg]                  # full chat history
    kbContext: Optional[str] = None      # optional: KB draft/snippets for THIS turn
    temperature: Optional[float] = 0.2   # quality-oriented defaults
    maxTokens: Optional[int] = 2048      # generous headroom; no output trimming

class ConverseReply(BaseModel):
    text: str

# ===== Streaming Converse models =====
class ConverseStreamBody(BaseModel):
    messages: List[Dict[str, Any]]        # same shape you send today: [{role, content:[{text}]}]
    kbContext: Optional[str] = None
    temperature: Optional[float] = 0.2
    maxTokens: Optional[int] = 2048

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
    br_runtime = bedrock_client  # reuse the same runtime client for Converse
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

@app.post("/chat/converse", response_model=ConverseReply)
def chat_converse(body: ConverseBody):
    if not MODEL_ID:
        raise HTTPException(status_code=500, detail="MODEL_ID is not configured.")

    # 1) System preamble: short, clear, quality-first
    system_preamble = (
        "You are HD Regulatory AI. Be precise, concise, and cite sources "
        "only if explicitly provided by the UI. If unsure, say so."
    )
    convo_msgs = [{
        "role": "system",
        "content": [{"text": system_preamble}]
    }]

    # 2) Optional KB grounding context for THIS turn (quality: include it fully if it’s short;
    #    if very long, allow up to a few thousand chars—it affects INPUT cost only, not output length)
    if body.kbContext:
        convo_msgs.append({
            "role": "system",
            "content": [{"text": f"Grounding context for this turn:\n{body.kbContext}"}]
        })

    # 3) Append chat history (messages[]) exactly as-is
    for m in body.messages:
        parts = [{"text": p.text} for p in (m.content or []) if isinstance(p.text, str)]
        if parts:
            convo_msgs.append({"role": m.role, "content": parts})

    # 4) Call Converse (quality-oriented defaults; no output clipping)
    try:
        resp = br_runtime.converse(
            modelId=MODEL_ID,
            messages=convo_msgs,
            inferenceConfig={
                "maxTokens": int(body.maxTokens or 2048),
                "temperature": float(body.temperature or 0.2),
                # You can add topP/topK if you ever need a different style
            },
            # requestOptions (timeouts) help production reliability without affecting quality
            requestOptions={"timeout": 60}  # seconds; adjust if needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Converse error: {str(e)}")

    # 5) Extract final assistant text
    out = ""
    try:
        msg = (resp.get("output") or {}).get("message") or {}
        parts = msg.get("content") or []
        out = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
    except Exception:
        out = ""

    return ConverseReply(text=out or "")

@app.post("/chat/converse/stream")
def chat_converse_stream(body: ConverseStreamBody):
    """
    Streaming version of Converse.
    Server-Sent Events (SSE) where each event is: {"type":"delta","text":"..."}
    Final events may include {"type":"usage", ...} and {"type":"done"}.
    """
    if not MODEL_ID:
        raise HTTPException(status_code=500, detail="MODEL_ID is not configured.")

    # 1) Build the messages exactly like /chat/converse (system preamble + optional KB context + history)
    system_preamble = (
        "You are HD Regulatory AI. Be precise, concise, and cite sources "
        "only if explicitly provided by the UI. If unsure, say so."
    )
    convo_msgs: List[Dict[str, Any]] = [{
        "role": "system",
        "content": [{"text": system_preamble}]
    }]

    if body.kbContext:
        convo_msgs.append({
            "role": "system",
            "content": [{"text": f"Grounding context for this turn:\n{body.kbContext}"}]
        })

    for m in body.messages:
        role = m.get("role")
        content = m.get("content") or []
        parts = [{"text": p.get("text", "")} for p in content if isinstance(p, dict) and p.get("text")]
        if role in ("user", "assistant", "system") and parts:
            convo_msgs.append({"role": role, "content": parts})

    # 2) Streaming generator -> SSE
    def sse_events():
        try:
            resp = br_runtime.converse_stream(
                modelId=MODEL_ID,
                messages=convo_msgs,
                inferenceConfig={
                    "maxTokens": int(body.maxTokens or 2048),
                    "temperature": float(body.temperature or 0.2),
                },
                requestOptions={"timeout": 60}
            )

            stream = resp.get("stream")
            if not stream:
                yield f"data: {json.dumps({'type':'error','error':'no stream'})}\n\n"
                return

            # Optional heartbeat to keep some proxies from closing idle SSE connections
            last_ping = time.time()

            for event in stream:
                # Content deltas
                cbd = event.get("contentBlockDelta")
                if cbd and isinstance(cbd, dict):
                    delta = (cbd.get("delta") or {}).get("text", "")
                    if delta:
                        yield f"data: {json.dumps({'type':'delta','text': delta})}\n\n"

                # Usage / metadata
                meta = event.get("metadata")
                if meta and isinstance(meta, dict):
                    usage = meta.get("usage") or {}
                    yield f"data: {json.dumps({'type':'usage','usage': usage})}\n\n"

                # Stream errors
                if ("internalServerException" in event
                    or "throttlingException" in event
                    or "modelStreamErrorException" in event):
                    yield f"data: {json.dumps({'type':'error','error':'stream exception','event':event})}\n\n"

                # Heartbeat every ~15s (SSE comment line)
                if time.time() - last_ping > 15:
                    yield ": keep-alive\n\n"
                    last_ping = time.time()

            # End of stream
            yield "data: {\"type\":\"done\"}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error': str(e)})}\n\n"

    # 3) Return an SSE StreamingResponse
    return StreamingResponse(
        sse_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# === END: POST variant for /bedrock/query ===

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)