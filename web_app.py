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

# CORS for frontends (e.g., file://, localhost:3000). Restrict later as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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