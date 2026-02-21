"""
Lamino - AI Chat Platform for Orcest AI Ecosystem
Full-featured chat with SSO auth, workspaces, file uploads, model selection,
decision chain tracking, and RainyModel auto-connect.
"""

import os
import uuid
import time
import json
import base64
import hashlib
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

router = APIRouter()

RAINYMODEL_BASE_URL = os.getenv("RAINYMODEL_BASE_URL", "https://rm.orcest.ai/v1")
RAINYMODEL_MASTER_KEY = os.getenv("RAINYMODEL_MASTER_KEY", "")
SSO_ISSUER = os.getenv("SSO_ISSUER", "https://login.orcest.ai")
SSO_CLIENT_ID = os.getenv("SSO_CLIENT_ID", "lamino")
SSO_CLIENT_SECRET = os.getenv("LAMINO_SSO_CLIENT_SECRET", os.getenv("SSO_CLIENT_SECRET", ""))
SSO_CALLBACK_URL = os.getenv("LAMINO_SSO_CALLBACK_URL", "https://llm.orcest.ai/auth/callback")
LAMINO_SSO_COOKIE = "lamino_sso_token"
LAMINO_USER_COOKIE = "lamino_user_info"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {
    "image": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"],
    "document": [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".md", ".csv"],
}

# In-memory storage (production would use database)
_workspaces: dict[str, dict] = {}
_chat_histories: dict[str, list] = {}
_uploaded_files: dict[str, dict] = {}
_user_settings: dict[str, dict] = {}


# --- Available Models & Providers (from orcest.ai ecosystem) ---

ORCEST_PROVIDERS = {
    "rainymodel": {
        "name": "RainyModel",
        "description": "Intelligent LLM Router (Default)",
        "base_url": RAINYMODEL_BASE_URL,
        "requires_key": False,
        "models": [
            {"id": "rainymodel/auto", "name": "RainyModel Auto", "description": "Cost-optimized automatic routing", "default": True},
            {"id": "rainymodel/chat", "name": "RainyModel Chat", "description": "Optimized for conversation"},
            {"id": "rainymodel/code", "name": "RainyModel Code", "description": "Code generation specialist"},
            {"id": "rainymodel/agent", "name": "RainyModel Agent", "description": "Complex agent tasks (Premium)"},
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": "Premium multi-provider gateway",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            {"id": "openrouter/qwen/qwen-2.5-72b-instruct", "name": "Qwen 2.5 72B"},
            {"id": "openrouter/anthropic/claude-sonnet-4", "name": "Claude Sonnet 4"},
            {"id": "openrouter/openai/gpt-4o", "name": "GPT-4o"},
            {"id": "openrouter/google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash"},
            {"id": "openrouter/meta-llama/llama-3.1-70b-instruct", "name": "LLaMA 3.1 70B"},
        ],
    },
    "ollamafreeapi": {
        "name": "OllamaFreeAPI",
        "description": "Free open-source models",
        "base_url": os.getenv("OLLAMAFREE_API_BASE", "https://ollamafreeapi.orcest.ai"),
        "requires_key": False,
        "env_key": "OLLAMAFREE_API_KEY",
        "models": [
            {"id": "llama3.1:8b", "name": "LLaMA 3.1 8B"},
            {"id": "qwen2.5:14b", "name": "Qwen 2.5 14B"},
            {"id": "mistral:7b", "name": "Mistral 7B"},
            {"id": "deepseek-r1:7b", "name": "DeepSeek R1 7B"},
        ],
    },
}


# --- SSO Auth Helpers ---

async def _verify_sso_token(token: str) -> dict | None:
    if not token:
        return None
    if not SSO_CLIENT_SECRET:
        return {"sub": "dev-user", "name": "Developer", "email": "dev@orcest.ai", "role": "admin"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"{SSO_ISSUER}/oauth2/userinfo",
                headers={"Authorization": f"Bearer {token}"},
            )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


async def _get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(LAMINO_SSO_COOKIE)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None
    return await _verify_sso_token(token)


def _sso_login_url(return_to: str = "/lamino") -> str:
    state_obj = {"returnTo": return_to}
    state_raw = json.dumps(state_obj).encode("utf-8")
    encoded = base64.urlsafe_b64encode(state_raw).decode("utf-8").rstrip("=")
    return (
        f"{SSO_ISSUER}/oauth2/authorize?client_id={SSO_CLIENT_ID}"
        f"&redirect_uri={SSO_CALLBACK_URL}&response_type=code"
        f"&scope=openid%20profile%20email&state={encoded}"
    )


def _user_id(user: dict) -> str:
    return user.get("sub", user.get("id", "anonymous"))


# --- SSO Auth Endpoints ---

@router.get("/auth/callback")
async def lamino_auth_callback(request: Request, code: str = "", state: str = ""):
    if not SSO_CLIENT_SECRET or not code:
        return RedirectResponse(url="/lamino", status_code=302)

    return_to = "/lamino"
    if state:
        try:
            decoded = base64.urlsafe_b64decode(state + "==")
            data = json.loads(decoded)
            return_to = data.get("returnTo", "/lamino")
        except Exception:
            pass

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            f"{SSO_ISSUER}/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": SSO_CALLBACK_URL,
                "client_id": SSO_CLIENT_ID,
                "client_secret": SSO_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if token_res.status_code != 200:
        return RedirectResponse(url=_sso_login_url(return_to), status_code=302)

    token_data = token_res.json()
    access_token = token_data.get("access_token")
    if not access_token:
        return RedirectResponse(url=_sso_login_url(return_to), status_code=302)

    redirect = RedirectResponse(url=return_to, status_code=302)
    redirect.set_cookie(
        key=LAMINO_SSO_COOKIE,
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,
    )
    return redirect


@router.get("/auth/logout")
async def lamino_logout():
    redirect = RedirectResponse(url=_sso_login_url(), status_code=302)
    redirect.delete_cookie(LAMINO_SSO_COOKIE)
    redirect.delete_cookie(LAMINO_USER_COOKIE)
    return redirect


# --- Workspace Management API ---

@router.get("/api/workspaces")
async def list_workspaces(request: Request):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    uid = _user_id(user)
    user_workspaces = [
        ws for ws in _workspaces.values()
        if ws["owner_id"] == uid or uid in ws.get("members", [])
    ]

    if not user_workspaces:
        default_ws = _create_default_workspace(uid, user.get("name", "User"))
        user_workspaces = [default_ws]

    return {"workspaces": user_workspaces}


@router.post("/api/workspaces")
async def create_workspace(request: Request):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    body = await request.json()
    uid = _user_id(user)
    ws_id = str(uuid.uuid4())
    workspace = {
        "id": ws_id,
        "name": body.get("name", "New Workspace"),
        "description": body.get("description", ""),
        "owner_id": uid,
        "owner_name": user.get("name", "User"),
        "members": body.get("members", []),
        "model": body.get("model", "rainymodel/auto"),
        "provider": body.get("provider", "rainymodel"),
        "system_prompt": body.get("system_prompt", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 4096),
            "policy": body.get("policy", "auto"),
        },
    }
    _workspaces[ws_id] = workspace
    _chat_histories[ws_id] = []
    return {"workspace": workspace}


@router.put("/api/workspaces/{ws_id}")
async def update_workspace(request: Request, ws_id: str):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    ws = _workspaces.get(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    uid = _user_id(user)
    if ws["owner_id"] != uid and uid not in ws.get("members", []):
        raise HTTPException(status_code=403, detail="Access denied")

    body = await request.json()
    for key in ["name", "description", "model", "provider", "system_prompt", "members"]:
        if key in body:
            ws[key] = body[key]
    if "settings" in body:
        ws["settings"].update(body["settings"])
    ws["updated_at"] = datetime.now(timezone.utc).isoformat()
    return {"workspace": ws}


@router.delete("/api/workspaces/{ws_id}")
async def delete_workspace(request: Request, ws_id: str):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    ws = _workspaces.get(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    uid = _user_id(user)
    if ws["owner_id"] != uid:
        raise HTTPException(status_code=403, detail="Only the owner can delete a workspace")

    del _workspaces[ws_id]
    _chat_histories.pop(ws_id, None)
    return {"deleted": True}


def _create_default_workspace(uid: str, name: str) -> dict:
    ws_id = hashlib.md5(f"default-{uid}".encode()).hexdigest()
    if ws_id in _workspaces:
        return _workspaces[ws_id]
    workspace = {
        "id": ws_id,
        "name": f"{name} - Default",
        "description": "Default workspace with RainyModel auto-connect",
        "owner_id": uid,
        "owner_name": name,
        "members": [],
        "model": "rainymodel/auto",
        "provider": "rainymodel",
        "system_prompt": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "policy": "auto",
        },
    }
    _workspaces[ws_id] = workspace
    _chat_histories[ws_id] = []
    return workspace


# --- Chat History API ---

@router.get("/api/workspaces/{ws_id}/history")
async def get_chat_history(request: Request, ws_id: str):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    ws = _workspaces.get(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    uid = _user_id(user)
    if ws["owner_id"] != uid and uid not in ws.get("members", []):
        raise HTTPException(status_code=403, detail="Access denied")

    return {"history": _chat_histories.get(ws_id, [])}


@router.delete("/api/workspaces/{ws_id}/history")
async def clear_chat_history(request: Request, ws_id: str):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    ws = _workspaces.get(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    uid = _user_id(user)
    if ws["owner_id"] != uid and uid not in ws.get("members", []):
        raise HTTPException(status_code=403, detail="Access denied")

    _chat_histories[ws_id] = []
    return {"cleared": True}


# --- Provider & Model API ---

@router.get("/api/providers")
async def list_providers(request: Request):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    providers = {}
    for key, provider in ORCEST_PROVIDERS.items():
        p = dict(provider)
        if p.get("requires_key"):
            env_key = p.get("env_key", "")
            p["key_configured"] = bool(os.getenv(env_key, ""))
        else:
            p["key_configured"] = True
        providers[key] = p

    return {"providers": providers}


@router.get("/api/models")
async def list_models(request: Request):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    all_models = []
    for provider_key, provider in ORCEST_PROVIDERS.items():
        for model in provider["models"]:
            m = dict(model)
            m["provider"] = provider_key
            m["provider_name"] = provider["name"]
            if provider.get("requires_key"):
                env_key = provider.get("env_key", "")
                m["available"] = bool(os.getenv(env_key, ""))
            else:
                m["available"] = True
            all_models.append(m)
    return {"models": all_models}


# --- File Upload API ---

@router.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...), workspace_id: str = Form("")):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    ext = os.path.splitext(file.filename or "")[1].lower()
    all_exts = ALLOWED_EXTENSIONS["image"] + ALLOWED_EXTENSIONS["document"]
    if ext not in all_exts:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    file_id = str(uuid.uuid4())
    file_type = "image" if ext in ALLOWED_EXTENSIONS["image"] else "document"

    file_record = {
        "id": file_id,
        "name": file.filename,
        "type": file_type,
        "extension": ext,
        "size": len(contents),
        "content_type": file.content_type or "application/octet-stream",
        "uploaded_by": _user_id(user),
        "workspace_id": workspace_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "data_b64": base64.b64encode(contents).decode("utf-8"),
    }
    _uploaded_files[file_id] = file_record

    return {
        "file": {
            "id": file_id,
            "name": file.filename,
            "type": file_type,
            "extension": ext,
            "size": len(contents),
        }
    }


@router.get("/api/files/{file_id}")
async def get_file(request: Request, file_id: str):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    f = _uploaded_files.get(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    data = base64.b64decode(f["data_b64"])
    return Response(
        content=data,
        media_type=f["content_type"],
        headers={"Content-Disposition": f'inline; filename="{f["name"]}"'},
    )


@router.get("/api/files/{file_id}/preview")
async def preview_file(request: Request, file_id: str):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    f = _uploaded_files.get(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file": {
            "id": f["id"],
            "name": f["name"],
            "type": f["type"],
            "extension": f["extension"],
            "size": f["size"],
            "is_image": f["type"] == "image",
            "data_url": f"data:{f['content_type']};base64,{f['data_b64']}" if f["type"] == "image" else None,
        }
    }


# --- Chat Completion API (proxies to RainyModel) ---

@router.post("/api/chat")
async def chat_completion(request: Request):
    user = await _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="SSO authentication required")

    body = await request.json()
    workspace_id = body.get("workspace_id", "")
    message = body.get("message", "")
    model = body.get("model", "rainymodel/auto")
    provider = body.get("provider", "rainymodel")
    policy = body.get("policy", "auto")
    attached_files = body.get("files", [])
    stream = body.get("stream", False)

    if not message and not attached_files:
        raise HTTPException(status_code=400, detail="Message or files required")

    ws = _workspaces.get(workspace_id) if workspace_id else None
    if ws:
        uid = _user_id(user)
        if ws["owner_id"] != uid and uid not in ws.get("members", []):
            raise HTTPException(status_code=403, detail="Access denied to this workspace")
        if not model or model == "rainymodel/auto":
            model = ws.get("model", "rainymodel/auto")
        if not provider or provider == "rainymodel":
            provider = ws.get("provider", "rainymodel")

    # Build messages array
    messages = []
    system_prompt = ""
    if ws and ws.get("system_prompt"):
        system_prompt = ws["system_prompt"]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add history
    history = _chat_histories.get(workspace_id, []) if workspace_id else []
    for h in history[-20:]:  # Last 20 messages for context
        messages.append({"role": h["role"], "content": h["content"]})

    # Build user message content
    user_content = message
    file_context = ""
    for fid in attached_files:
        f = _uploaded_files.get(fid)
        if f and f["type"] == "document":
            file_context += f"\n[Attached file: {f['name']} ({f['extension']})]\n"

    if file_context:
        user_content = f"{message}\n\n--- Attached Files ---{file_context}"

    messages.append({"role": "user", "content": user_content})

    # Store user message in history
    user_msg_record = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": message,
        "files": attached_files,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
    }
    if workspace_id and workspace_id in _chat_histories:
        _chat_histories[workspace_id].append(user_msg_record)

    # Determine provider endpoint
    provider_config = ORCEST_PROVIDERS.get(provider, ORCEST_PROVIDERS["rainymodel"])
    api_base = provider_config["base_url"]
    api_key = RAINYMODEL_MASTER_KEY

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    elif provider == "ollamafreeapi":
        api_key = os.getenv("OLLAMAFREE_API_KEY", "")

    # Build request
    settings = ws.get("settings", {}) if ws else {}
    chat_request = {
        "model": model,
        "messages": messages,
        "temperature": settings.get("temperature", 0.7),
        "max_tokens": settings.get("max_tokens", 4096),
        "stream": stream,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-RainyModel-Policy": policy,
    }

    request_start = time.time()

    if stream:
        return StreamingResponse(
            _stream_chat_response(
                api_base, headers, chat_request, workspace_id,
                model, provider, policy, request_start
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json=chat_request,
                headers=headers,
            )

        latency_ms = int((time.time() - request_start) * 1000)

        if resp.status_code != 200:
            error_detail = resp.text[:500]
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": error_detail, "provider": provider, "model": model},
            )

        resp_data = resp.json()
        assistant_content = ""
        if resp_data.get("choices"):
            assistant_content = resp_data["choices"][0].get("message", {}).get("content", "")

        # Extract routing metadata from headers
        route = resp.headers.get("x-rainymodel-route", provider)
        upstream = resp.headers.get("x-rainymodel-upstream", provider)
        actual_model = resp.headers.get("x-rainymodel-model", model)
        rm_latency = resp.headers.get("x-rainymodel-latency-ms", str(latency_ms))
        fallback_reason = resp.headers.get("x-rainymodel-fallback-reason", "")

        # Build decision chain
        decision_chain = _build_decision_chain(
            provider, model, route, upstream, actual_model, policy, fallback_reason
        )

        # Store assistant message
        assistant_record = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": assistant_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "actual_model": actual_model,
            "provider": provider,
            "route": route,
            "upstream": upstream,
            "latency_ms": latency_ms,
            "decision_chain": decision_chain,
            "usage": resp_data.get("usage", {}),
        }

        if workspace_id and workspace_id in _chat_histories:
            _chat_histories[workspace_id].append(assistant_record)

        return {
            "response": assistant_content,
            "metadata": {
                "model_requested": model,
                "model_actual": actual_model,
                "provider": provider,
                "route": route,
                "upstream": upstream,
                "latency_ms": latency_ms,
                "policy": policy,
                "fallback_reason": fallback_reason,
                "decision_chain": decision_chain,
                "usage": resp_data.get("usage", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": "Request timed out", "provider": provider, "model": model},
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": str(e), "provider": provider, "model": model},
        )


async def _stream_chat_response(
    api_base: str,
    headers: dict,
    chat_request: dict,
    workspace_id: str,
    model: str,
    provider: str,
    policy: str,
    request_start: float,
):
    full_content = ""
    route = provider
    upstream = provider
    actual_model = model
    fallback_reason = ""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{api_base}/chat/completions",
                json=chat_request,
                headers=headers,
            ) as resp:
                route = resp.headers.get("x-rainymodel-route", provider)
                upstream = resp.headers.get("x-rainymodel-upstream", provider)
                actual_model = resp.headers.get("x-rainymodel-model", model)
                fallback_reason = resp.headers.get("x-rainymodel-fallback-reason", "")

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    latency_ms = int((time.time() - request_start) * 1000)
    decision_chain = _build_decision_chain(
        provider, model, route, upstream, actual_model, policy, fallback_reason
    )

    # Send metadata as final event
    metadata = {
        "model_requested": model,
        "model_actual": actual_model,
        "provider": provider,
        "route": route,
        "upstream": upstream,
        "latency_ms": latency_ms,
        "policy": policy,
        "fallback_reason": fallback_reason,
        "decision_chain": decision_chain,
    }
    yield f"data: {json.dumps({'metadata': metadata})}\n\n"
    yield "data: [DONE]\n\n"

    # Store in history
    if workspace_id and workspace_id in _chat_histories:
        _chat_histories[workspace_id].append({
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": full_content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "actual_model": actual_model,
            "provider": provider,
            "route": route,
            "upstream": upstream,
            "latency_ms": latency_ms,
            "decision_chain": decision_chain,
        })


def _build_decision_chain(
    provider: str,
    model_requested: str,
    route: str,
    upstream: str,
    actual_model: str,
    policy: str,
    fallback_reason: str,
) -> list[dict]:
    chain = []

    # Step 1: User request
    chain.append({
        "step": 1,
        "node": "User Request",
        "detail": f"Model: {model_requested}, Policy: {policy}",
        "type": "request",
    })

    # Step 2: RainyModel routing (if applicable)
    if provider == "rainymodel" or "rainymodel" in model_requested:
        chain.append({
            "step": 2,
            "node": "RainyModel Router",
            "detail": f"Policy: {policy}, Route: {route}",
            "type": "router",
        })

        # Step 3: Upstream selection
        upstream_name = {
            "hf": "HuggingFace Router",
            "ollama": "Ollama (Self-hosted)",
            "openrouter": "OpenRouter (Premium)",
            "ollamafreeapi": "OllamaFreeAPI (Free)",
        }.get(upstream, upstream)

        chain.append({
            "step": 3,
            "node": upstream_name,
            "detail": f"Upstream: {upstream}" + (f", Fallback: {fallback_reason}" if fallback_reason else ""),
            "type": "upstream",
        })
    else:
        chain.append({
            "step": 2,
            "node": ORCEST_PROVIDERS.get(provider, {}).get("name", provider),
            "detail": f"Direct provider call",
            "type": "provider",
        })

    # Final step: Actual model
    chain.append({
        "step": len(chain) + 1,
        "node": f"Model: {actual_model}",
        "detail": f"Final model execution",
        "type": "model",
    })

    return chain


# --- Health Check ---

@router.get("/api/health")
async def lamino_health():
    return {
        "status": "healthy",
        "service": "lamino",
        "version": "2.0.0",
        "features": [
            "sso_only_auth",
            "per_user_workspaces",
            "team_enterprise_premium",
            "rainymodel_auto_connect",
            "provider_model_selection",
            "decision_chain_tracking",
            "file_upload",
            "langchain_integration",
        ],
    }


# --- Main Chat UI ---

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def lamino_chat_page(request: Request):
    user = await _get_current_user(request)
    if not user:
        return RedirectResponse(url=_sso_login_url("/lamino"), status_code=302)
    return HTMLResponse(content=_lamino_chat_html(user))


def _lamino_chat_html(user: dict) -> str:
    user_name = user.get("name", "User")
    user_email = user.get("email", "")
    user_role = user.get("role", "viewer")
    user_json = json.dumps({"name": user_name, "email": user_email, "role": user_role, "id": _user_id(user)})

    return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lamino - Orcest AI Chat</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#0a0a0f;--bg2:#111827;--bg3:#1f2937;--bg4:#374151;
  --text:#f8fafc;--text2:#e2e8f0;--muted:#94a3b8;
  --blue:#60a5fa;--purple:#a78bfa;--green:#4ade80;--yellow:#fbbf24;
  --red:#ef4444;--cyan:#38bdf8;--border:#374151;
  --msg-user:#1e3a5f;--msg-ai:#1a1a2e;
}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text2);height:100vh;overflow:hidden;display:flex}}

/* Sidebar */
.sidebar{{width:280px;background:var(--bg2);border-left:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}}
.sidebar-header{{padding:16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}}
.sidebar-header h2{{font-size:1.1rem;background:linear-gradient(135deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.new-ws-btn{{background:var(--blue);color:white;border:none;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:.85rem}}
.new-ws-btn:hover{{opacity:.9}}
.ws-list{{flex:1;overflow-y:auto;padding:8px}}
.ws-item{{padding:12px;border-radius:10px;cursor:pointer;margin-bottom:4px;transition:background .2s;border:1px solid transparent}}
.ws-item:hover,.ws-item.active{{background:var(--bg3);border-color:var(--border)}}
.ws-item .ws-name{{font-weight:600;font-size:.95rem;color:var(--text);margin-bottom:2px}}
.ws-item .ws-model{{font-size:.8rem;color:var(--muted)}}
.sidebar-footer{{padding:12px;border-top:1px solid var(--border)}}
.user-info{{display:flex;align-items:center;gap:10px}}
.user-avatar{{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:.9rem}}
.user-details{{flex:1}}
.user-details .name{{font-size:.9rem;font-weight:600;color:var(--text)}}
.user-details .role{{font-size:.75rem;color:var(--muted);text-transform:capitalize}}
.logout-btn{{background:none;border:none;color:var(--red);cursor:pointer;font-size:.8rem;padding:4px 8px;border-radius:4px}}
.logout-btn:hover{{background:rgba(239,68,68,.1)}}

/* Main Chat Area */
.main{{flex:1;display:flex;flex-direction:column;min-width:0}}
.chat-header{{padding:12px 20px;border-bottom:1px solid var(--border);background:var(--bg2);display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
.chat-header-info{{display:flex;align-items:center;gap:12px}}
.chat-header-info h3{{font-size:1rem;color:var(--text)}}
.model-badge{{background:var(--bg3);color:var(--cyan);padding:4px 10px;border-radius:12px;font-size:.78rem;font-weight:600}}
.header-actions{{display:flex;gap:8px}}
.header-actions button{{background:var(--bg3);color:var(--text2);border:1px solid var(--border);border-radius:8px;padding:6px 12px;cursor:pointer;font-size:.82rem}}
.header-actions button:hover{{border-color:var(--blue);color:var(--blue)}}

/* Messages */
.messages{{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}}
.message{{max-width:85%;padding:14px 18px;border-radius:16px;line-height:1.7;position:relative}}
.message.user{{background:var(--msg-user);align-self:flex-end;border-bottom-left-radius:4px}}
.message.assistant{{background:var(--msg-ai);align-self:flex-start;border-bottom-right-radius:4px;border:1px solid var(--border)}}
.message .content{{white-space:pre-wrap;word-break:break-word}}
.message .content code{{background:rgba(255,255,255,.1);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:.9em}}
.message .content pre{{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin:8px 0;overflow-x:auto}}
.message .content pre code{{background:none;padding:0}}
.message .files{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
.message .file-badge{{background:var(--bg3);padding:4px 10px;border-radius:6px;font-size:.78rem;color:var(--muted)}}
.message .file-preview{{max-width:300px;max-height:200px;border-radius:8px;margin-top:8px}}

/* Prompt Details (shown below assistant messages) */
.prompt-details{{margin-top:10px;padding:10px 14px;background:var(--bg);border:1px solid var(--border);border-radius:10px;font-size:.8rem}}
.prompt-details summary{{color:var(--muted);cursor:pointer;font-weight:600;margin-bottom:6px}}
.prompt-details summary:hover{{color:var(--blue)}}
.detail-row{{display:flex;justify-content:space-between;padding:3px 0;color:var(--muted)}}
.detail-row .label{{color:var(--text2)}}
.detail-row .value{{color:var(--cyan);font-family:monospace;font-size:.82rem}}
.decision-chain{{margin-top:8px;padding:8px;background:var(--bg2);border-radius:8px}}
.chain-step{{display:flex;align-items:center;gap:8px;padding:4px 0}}
.chain-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.chain-dot.request{{background:var(--blue)}}
.chain-dot.router{{background:var(--purple)}}
.chain-dot.upstream{{background:var(--yellow)}}
.chain-dot.provider{{background:var(--green)}}
.chain-dot.model{{background:var(--cyan)}}
.chain-arrow{{color:var(--muted);font-size:.7rem}}
.chain-node{{font-weight:600;color:var(--text)}}
.chain-detail{{color:var(--muted);font-size:.78rem}}

/* Input Area */
.input-area{{padding:16px 20px;border-top:1px solid var(--border);background:var(--bg2);flex-shrink:0}}
.input-row{{display:flex;gap:8px;align-items:flex-end}}
.input-wrapper{{flex:1;position:relative}}
.input-wrapper textarea{{width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:12px;padding:12px 16px;font-size:.95rem;resize:none;outline:none;font-family:inherit;min-height:48px;max-height:200px;direction:rtl}}
.input-wrapper textarea:focus{{border-color:var(--blue)}}
.input-wrapper textarea::placeholder{{color:var(--muted)}}
.input-actions{{display:flex;gap:6px}}
.input-actions button{{background:var(--bg3);color:var(--text2);border:1px solid var(--border);border-radius:10px;width:44px;height:44px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:1.1rem;transition:all .2s}}
.input-actions button:hover{{border-color:var(--blue);color:var(--blue)}}
.send-btn{{background:linear-gradient(135deg,var(--blue),var(--purple))!important;color:white!important;border:none!important}}
.send-btn:hover{{opacity:.9!important}}
.send-btn:disabled{{opacity:.5!important;cursor:not-allowed!important}}
.attached-files{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}}
.attached-file{{background:var(--bg3);padding:4px 10px;border-radius:8px;font-size:.8rem;color:var(--text2);display:flex;align-items:center;gap:6px}}
.attached-file .remove{{cursor:pointer;color:var(--red);font-weight:bold}}

/* Model Selector */
.model-selector-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;justify-content:center;align-items:center}}
.model-selector-overlay.show{{display:flex}}
.model-selector{{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:600px;max-height:80vh;overflow-y:auto}}
.model-selector h3{{color:var(--text);margin-bottom:16px}}
.provider-group{{margin-bottom:16px}}
.provider-group h4{{color:var(--muted);font-size:.85rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}}
.model-option{{display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:10px;cursor:pointer;border:1px solid transparent;transition:all .2s}}
.model-option:hover{{background:var(--bg3);border-color:var(--border)}}
.model-option.selected{{background:rgba(96,165,250,.1);border-color:var(--blue)}}
.model-option .model-name{{font-weight:600;color:var(--text)}}
.model-option .model-desc{{font-size:.82rem;color:var(--muted)}}
.model-option .unavailable{{color:var(--red);font-size:.78rem}}

/* Workspace Settings */
.ws-settings-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;justify-content:center;align-items:center}}
.ws-settings-overlay.show{{display:flex}}
.ws-settings{{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:24px;width:90%;max-width:500px}}
.ws-settings h3{{color:var(--text);margin-bottom:16px}}
.form-group{{margin-bottom:14px}}
.form-group label{{display:block;color:var(--text2);font-size:.85rem;margin-bottom:4px;font-weight:600}}
.form-group input,.form-group textarea,.form-group select{{width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:10px;font-size:.9rem;outline:none;font-family:inherit}}
.form-group input:focus,.form-group textarea:focus,.form-group select:focus{{border-color:var(--blue)}}
.form-actions{{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}}
.form-actions button{{padding:8px 20px;border-radius:8px;cursor:pointer;font-size:.9rem;border:none}}
.btn-save{{background:var(--blue);color:white}}
.btn-cancel{{background:var(--bg3);color:var(--text2);border:1px solid var(--border)!important}}
.btn-danger{{background:var(--red);color:white}}

/* Loading spinner */
.typing-indicator{{display:flex;gap:4px;padding:8px 12px}}
.typing-dot{{width:8px;height:8px;background:var(--muted);border-radius:50%;animation:typing 1.2s infinite}}
.typing-dot:nth-child(2){{animation-delay:.2s}}
.typing-dot:nth-child(3){{animation-delay:.4s}}
@keyframes typing{{0%,80%,100%{{opacity:.3;transform:scale(.8)}}40%{{opacity:1;transform:scale(1)}}}}

/* Scrollbar */
::-webkit-scrollbar{{width:6px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--bg4);border-radius:3px}}

/* Responsive */
@media(max-width:768px){{
  .sidebar{{position:fixed;right:-280px;top:0;bottom:0;z-index:50;transition:right .3s}}
  .sidebar.open{{right:0}}
  .sidebar-toggle{{display:block!important}}
}}
.sidebar-toggle{{display:none;position:fixed;top:12px;right:12px;z-index:60;background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:8px 12px;cursor:pointer;color:var(--text)}}

/* Empty state */
.empty-state{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);gap:16px}}
.empty-state h2{{color:var(--text);font-size:1.5rem}}
.empty-state p{{max-width:400px;text-align:center;line-height:1.6}}
</style>
</head>
<body>
<button class="sidebar-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">&#9776;</button>

<!-- Sidebar -->
<aside class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <h2>Lamino</h2>
    <button class="new-ws-btn" onclick="createWorkspace()">+ Workspace</button>
  </div>
  <div class="ws-list" id="wsList"></div>
  <div class="sidebar-footer">
    <div class="user-info">
      <div class="user-avatar">{user_name[0].upper() if user_name else "U"}</div>
      <div class="user-details">
        <div class="name">{user_name}</div>
        <div class="role">{user_role}</div>
      </div>
      <button class="logout-btn" onclick="location.href='/lamino/auth/logout'">Exit</button>
    </div>
  </div>
</aside>

<!-- Main Chat -->
<main class="main">
  <div class="chat-header" id="chatHeader">
    <div class="chat-header-info">
      <h3 id="wsTitle">Select a Workspace</h3>
      <span class="model-badge" id="modelBadge">rainymodel/auto</span>
    </div>
    <div class="header-actions">
      <button onclick="openModelSelector()">Change Model</button>
      <button onclick="openWsSettings()">Settings</button>
      <button onclick="clearHistory()">Clear</button>
    </div>
  </div>

  <div class="messages" id="messages">
    <div class="empty-state" id="emptyState">
      <h2>Welcome to Lamino</h2>
      <p>AI Chat Platform by Orcest AI. Connected to RainyModel for intelligent routing. Start a conversation or select a workspace.</p>
    </div>
  </div>

  <div class="input-area">
    <div class="attached-files" id="attachedFiles"></div>
    <div class="input-row">
      <div class="input-actions">
        <button onclick="document.getElementById('fileInput').click()" title="Upload file">&#128206;</button>
        <input type="file" id="fileInput" style="display:none" accept=".png,.jpg,.jpeg,.gif,.webp,.pdf,.doc,.docx,.ppt,.pptx,.txt,.md,.csv" multiple onchange="handleFileUpload(this)">
      </div>
      <div class="input-wrapper">
        <textarea id="messageInput" placeholder="Type your message..." rows="1" onkeydown="handleKeyDown(event)" oninput="autoResize(this)"></textarea>
      </div>
      <div class="input-actions">
        <button class="send-btn" id="sendBtn" onclick="sendMessage()" disabled>&#10148;</button>
      </div>
    </div>
  </div>
</main>

<!-- Model Selector Modal -->
<div class="model-selector-overlay" id="modelSelectorOverlay" onclick="if(event.target===this)this.classList.remove('show')">
  <div class="model-selector" id="modelSelectorContent"></div>
</div>

<!-- Workspace Settings Modal -->
<div class="ws-settings-overlay" id="wsSettingsOverlay" onclick="if(event.target===this)this.classList.remove('show')">
  <div class="ws-settings" id="wsSettingsContent"></div>
</div>

<script>
const API = '/lamino/api';
const currentUser = {user_json};
let workspaces = [];
let activeWs = null;
let attachedFileIds = [];
let isStreaming = false;

// --- Init ---
async function init() {{
  await loadWorkspaces();
  document.getElementById('messageInput').addEventListener('input', () => {{
    document.getElementById('sendBtn').disabled = !document.getElementById('messageInput').value.trim() && attachedFileIds.length === 0;
  }});
}}

// --- Workspaces ---
async function loadWorkspaces() {{
  try {{
    const res = await fetch(API + '/workspaces');
    const data = await res.json();
    workspaces = data.workspaces || [];
    renderWorkspaces();
    if (workspaces.length > 0 && !activeWs) {{
      selectWorkspace(workspaces[0].id);
    }}
  }} catch (e) {{
    console.error('Failed to load workspaces:', e);
  }}
}}

function renderWorkspaces() {{
  const list = document.getElementById('wsList');
  list.innerHTML = workspaces.map(ws => `
    <div class="ws-item ${{activeWs && activeWs.id === ws.id ? 'active' : ''}}" onclick="selectWorkspace('${{ws.id}}')">
      <div class="ws-name">${{ws.name}}</div>
      <div class="ws-model">${{ws.model || 'rainymodel/auto'}}</div>
    </div>
  `).join('');
}}

async function selectWorkspace(wsId) {{
  activeWs = workspaces.find(w => w.id === wsId);
  if (!activeWs) return;
  renderWorkspaces();
  document.getElementById('wsTitle').textContent = activeWs.name;
  document.getElementById('modelBadge').textContent = activeWs.model || 'rainymodel/auto';
  await loadHistory();
}}

async function createWorkspace() {{
  const name = prompt('Workspace name:');
  if (!name) return;
  try {{
    const res = await fetch(API + '/workspaces', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{name, model: 'rainymodel/auto', provider: 'rainymodel'}}),
    }});
    const data = await res.json();
    workspaces.push(data.workspace);
    selectWorkspace(data.workspace.id);
    renderWorkspaces();
  }} catch (e) {{
    alert('Failed to create workspace');
  }}
}}

// --- Chat History ---
async function loadHistory() {{
  if (!activeWs) return;
  try {{
    const res = await fetch(API + '/workspaces/' + activeWs.id + '/history');
    const data = await res.json();
    renderMessages(data.history || []);
  }} catch (e) {{
    console.error('Failed to load history:', e);
  }}
}}

function renderMessages(history) {{
  const container = document.getElementById('messages');
  const empty = document.getElementById('emptyState');
  if (history.length === 0) {{
    container.innerHTML = '';
    container.appendChild(empty);
    empty.style.display = 'flex';
    return;
  }}
  empty.style.display = 'none';
  container.innerHTML = history.map(msg => renderMessage(msg)).join('');
  container.scrollTop = container.scrollHeight;
}}

function renderMessage(msg) {{
  let filesHtml = '';
  if (msg.files && msg.files.length > 0) {{
    filesHtml = '<div class="files">' + msg.files.map(fid => `<span class="file-badge">&#128196; File</span>`).join('') + '</div>';
  }}

  let detailsHtml = '';
  if (msg.role === 'assistant' && msg.decision_chain) {{
    detailsHtml = renderPromptDetails(msg);
  }}

  return `<div class="message ${{msg.role}}">
    <div class="content">${{escapeHtml(msg.content || '')}}</div>
    ${{filesHtml}}
    ${{detailsHtml}}
  </div>`;
}}

function renderPromptDetails(msg) {{
  const chain = msg.decision_chain || [];
  const chainHtml = chain.map(step => `
    <div class="chain-step">
      <span class="chain-dot ${{step.type}}"></span>
      <span class="chain-node">${{step.node}}</span>
      <span class="chain-arrow">&#8594;</span>
      <span class="chain-detail">${{step.detail}}</span>
    </div>
  `).join('');

  return `<details class="prompt-details">
    <summary>Prompt Details & Decision Chain</summary>
    <div class="detail-row"><span class="label">Requested Model:</span><span class="value">${{msg.model || '-'}}</span></div>
    <div class="detail-row"><span class="label">Actual Model:</span><span class="value">${{msg.actual_model || msg.model || '-'}}</span></div>
    <div class="detail-row"><span class="label">Provider:</span><span class="value">${{msg.provider || '-'}}</span></div>
    <div class="detail-row"><span class="label">Route:</span><span class="value">${{msg.route || '-'}}</span></div>
    <div class="detail-row"><span class="label">Upstream:</span><span class="value">${{msg.upstream || '-'}}</span></div>
    <div class="detail-row"><span class="label">Latency:</span><span class="value">${{msg.latency_ms || 0}}ms</span></div>
    ${{chain.length > 0 ? '<div class="decision-chain"><strong style="color:var(--muted);font-size:.82rem">Decision Chain:</strong>' + chainHtml + '</div>' : ''}}
  </details>`;
}}

// --- Send Message ---
async function sendMessage() {{
  const input = document.getElementById('messageInput');
  const message = input.value.trim();
  if (!message && attachedFileIds.length === 0) return;
  if (!activeWs) {{
    alert('Please select or create a workspace first');
    return;
  }}

  // Add user message to UI
  const msgContainer = document.getElementById('messages');
  document.getElementById('emptyState').style.display = 'none';

  const userMsgHtml = `<div class="message user"><div class="content">${{escapeHtml(message)}}</div></div>`;
  msgContainer.insertAdjacentHTML('beforeend', userMsgHtml);

  input.value = '';
  input.style.height = '48px';
  document.getElementById('sendBtn').disabled = true;

  // Add typing indicator
  const typingId = 'typing-' + Date.now();
  msgContainer.insertAdjacentHTML('beforeend', `<div class="message assistant" id="${{typingId}}"><div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div></div>`);
  msgContainer.scrollTop = msgContainer.scrollHeight;

  const files = [...attachedFileIds];
  clearAttachedFiles();

  try {{
    const res = await fetch(API + '/chat', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        workspace_id: activeWs.id,
        message,
        model: activeWs.model || 'rainymodel/auto',
        provider: activeWs.provider || 'rainymodel',
        policy: (activeWs.settings || {{}}).policy || 'auto',
        files,
        stream: false,
      }}),
    }});

    const data = await res.json();
    const typing = document.getElementById(typingId);
    if (typing) typing.remove();

    if (data.error) {{
      msgContainer.insertAdjacentHTML('beforeend', `<div class="message assistant"><div class="content" style="color:var(--red)">Error: ${{escapeHtml(data.error)}}</div></div>`);
    }} else {{
      const assistantMsg = {{
        role: 'assistant',
        content: data.response,
        model: data.metadata?.model_requested,
        actual_model: data.metadata?.model_actual,
        provider: data.metadata?.provider,
        route: data.metadata?.route,
        upstream: data.metadata?.upstream,
        latency_ms: data.metadata?.latency_ms,
        decision_chain: data.metadata?.decision_chain,
      }};
      msgContainer.insertAdjacentHTML('beforeend', renderMessage(assistantMsg));
    }}
    msgContainer.scrollTop = msgContainer.scrollHeight;
  }} catch (e) {{
    const typing = document.getElementById(typingId);
    if (typing) typing.remove();
    msgContainer.insertAdjacentHTML('beforeend', `<div class="message assistant"><div class="content" style="color:var(--red)">Connection error: ${{e.message}}</div></div>`);
    msgContainer.scrollTop = msgContainer.scrollHeight;
  }}
}}

// --- File Upload ---
async function handleFileUpload(input) {{
  const files = input.files;
  for (const file of files) {{
    const formData = new FormData();
    formData.append('file', file);
    formData.append('workspace_id', activeWs ? activeWs.id : '');
    try {{
      const res = await fetch(API + '/upload', {{method: 'POST', body: formData}});
      const data = await res.json();
      if (data.file) {{
        attachedFileIds.push(data.file.id);
        renderAttachedFiles(data.file);
      }}
    }} catch (e) {{
      alert('Upload failed: ' + e.message);
    }}
  }}
  input.value = '';
  document.getElementById('sendBtn').disabled = !document.getElementById('messageInput').value.trim() && attachedFileIds.length === 0;
}}

function renderAttachedFiles(file) {{
  const container = document.getElementById('attachedFiles');
  container.insertAdjacentHTML('beforeend', `<div class="attached-file" id="af-${{file.id}}">&#128196; ${{file.name}} <span class="remove" onclick="removeAttachedFile('${{file.id}}')">&times;</span></div>`);
}}

function removeAttachedFile(fid) {{
  attachedFileIds = attachedFileIds.filter(id => id !== fid);
  const el = document.getElementById('af-' + fid);
  if (el) el.remove();
  document.getElementById('sendBtn').disabled = !document.getElementById('messageInput').value.trim() && attachedFileIds.length === 0;
}}

function clearAttachedFiles() {{
  attachedFileIds = [];
  document.getElementById('attachedFiles').innerHTML = '';
}}

// --- Model Selector ---
async function openModelSelector() {{
  try {{
    const res = await fetch(API + '/providers');
    const data = await res.json();
    const providers = data.providers || {{}};
    let html = '<h3>Select Model</h3>';

    for (const [key, provider] of Object.entries(providers)) {{
      html += `<div class="provider-group"><h4>${{provider.name}} ${{!provider.key_configured ? '<span style="color:var(--red)">(Key not configured)</span>' : ''}}</h4>`;
      for (const model of provider.models || []) {{
        const isSelected = activeWs && activeWs.model === model.id;
        const isAvailable = provider.key_configured;
        html += `<div class="model-option ${{isSelected ? 'selected' : ''}} ${{!isAvailable ? 'unavailable' : ''}}" onclick="${{isAvailable ? `selectModel('${{key}}','${{model.id}}')` : ''}}">
          <div>
            <div class="model-name">${{model.name}}${{model.default ? ' (Default)' : ''}}</div>
            <div class="model-desc">${{model.description || ''}}</div>
            ${{!isAvailable ? '<div class="unavailable">API key not configured</div>' : ''}}
          </div>
        </div>`;
      }}
      html += '</div>';
    }}
    html += '<div style="margin-top:16px;text-align:left"><button onclick="document.getElementById(\\'modelSelectorOverlay\\').classList.remove(\\'show\\')" style="background:var(--bg3);color:var(--text2);border:1px solid var(--border);border-radius:8px;padding:8px 20px;cursor:pointer">Close</button></div>';

    document.getElementById('modelSelectorContent').innerHTML = html;
    document.getElementById('modelSelectorOverlay').classList.add('show');
  }} catch (e) {{
    alert('Failed to load models');
  }}
}}

async function selectModel(provider, modelId) {{
  if (!activeWs) return;
  try {{
    await fetch(API + '/workspaces/' + activeWs.id, {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{model: modelId, provider}}),
    }});
    activeWs.model = modelId;
    activeWs.provider = provider;
    document.getElementById('modelBadge').textContent = modelId;
    document.getElementById('modelSelectorOverlay').classList.remove('show');
    renderWorkspaces();
  }} catch (e) {{
    alert('Failed to update model');
  }}
}}

// --- Workspace Settings ---
function openWsSettings() {{
  if (!activeWs) return;
  const ws = activeWs;
  const html = `<h3>Workspace Settings</h3>
    <div class="form-group"><label>Name</label><input id="wsName" value="${{ws.name}}"></div>
    <div class="form-group"><label>Description</label><textarea id="wsDesc" rows="2">${{ws.description || ''}}</textarea></div>
    <div class="form-group"><label>System Prompt</label><textarea id="wsSysPrompt" rows="3" placeholder="Optional system prompt for this workspace...">${{ws.system_prompt || ''}}</textarea></div>
    <div class="form-group"><label>Policy</label>
      <select id="wsPolicy">
        <option value="auto" ${{(ws.settings||{{}}).policy==='auto'?'selected':''}}>Auto (Cost-optimized)</option>
        <option value="premium" ${{(ws.settings||{{}}).policy==='premium'?'selected':''}}>Premium (Best quality)</option>
        <option value="free" ${{(ws.settings||{{}}).policy==='free'?'selected':''}}>Free (No premium)</option>
        <option value="uncensored" ${{(ws.settings||{{}}).policy==='uncensored'?'selected':''}}>Uncensored (Internal first)</option>
      </select>
    </div>
    <div class="form-group"><label>Temperature (${{(ws.settings||{{}}).temperature||0.7}})</label><input type="range" id="wsTemp" min="0" max="2" step="0.1" value="${{(ws.settings||{{}}).temperature||0.7}}" oninput="this.previousElementSibling.textContent='Temperature ('+this.value+')'"></div>
    <div class="form-group"><label>Max Tokens</label><input type="number" id="wsMaxTokens" value="${{(ws.settings||{{}}).max_tokens||4096}}" min="128" max="32768"></div>
    <div class="form-group"><label>Team Members (comma-separated user IDs)</label><input id="wsMembers" value="${{(ws.members||[]).join(',')}}"></div>
    <div class="form-actions">
      <button class="btn-danger" onclick="deleteWorkspace()">Delete</button>
      <button class="btn-cancel" onclick="document.getElementById('wsSettingsOverlay').classList.remove('show')">Cancel</button>
      <button class="btn-save" onclick="saveWsSettings()">Save</button>
    </div>`;
  document.getElementById('wsSettingsContent').innerHTML = html;
  document.getElementById('wsSettingsOverlay').classList.add('show');
}}

async function saveWsSettings() {{
  if (!activeWs) return;
  const members = document.getElementById('wsMembers').value.split(',').map(s => s.trim()).filter(Boolean);
  try {{
    await fetch(API + '/workspaces/' + activeWs.id, {{
      method: 'PUT',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        name: document.getElementById('wsName').value,
        description: document.getElementById('wsDesc').value,
        system_prompt: document.getElementById('wsSysPrompt').value,
        members,
        settings: {{
          policy: document.getElementById('wsPolicy').value,
          temperature: parseFloat(document.getElementById('wsTemp').value),
          max_tokens: parseInt(document.getElementById('wsMaxTokens').value),
        }},
      }}),
    }});
    document.getElementById('wsSettingsOverlay').classList.remove('show');
    await loadWorkspaces();
    selectWorkspace(activeWs.id);
  }} catch (e) {{
    alert('Failed to save settings');
  }}
}}

async function deleteWorkspace() {{
  if (!activeWs) return;
  if (!confirm('Are you sure you want to delete this workspace?')) return;
  try {{
    await fetch(API + '/workspaces/' + activeWs.id, {{method: 'DELETE'}});
    activeWs = null;
    await loadWorkspaces();
    document.getElementById('wsSettingsOverlay').classList.remove('show');
  }} catch (e) {{
    alert('Failed to delete workspace');
  }}
}}

async function clearHistory() {{
  if (!activeWs) return;
  if (!confirm('Clear all messages in this workspace?')) return;
  try {{
    await fetch(API + '/workspaces/' + activeWs.id + '/history', {{method: 'DELETE'}});
    await loadHistory();
  }} catch (e) {{
    alert('Failed to clear history');
  }}
}}

// --- Utilities ---
function escapeHtml(str) {{
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}}

function autoResize(el) {{
  el.style.height = '48px';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}}

function handleKeyDown(e) {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
}}

init();
</script>
</body>
</html>"""
