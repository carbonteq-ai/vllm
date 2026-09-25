# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session lifecycle hints for the prefix cache (CarbonTeq)."""

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient

router = APIRouter()


class ReleaseSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/v1/sessions/release")
async def release_session(body: ReleaseSessionRequest, raw_request: Request):
    """Tell the server a session will send no more requests.

    The session is the one named by ``session_id`` in request bodies or the
    ``X-Session-ID`` header. Its cached prefix becomes the first to be evicted,
    except blocks shared with sessions still running. It stays cached until
    then, so a late request still hits it.

    Returns ``{"released_blocks": int}``.
    """
    released = await engine_client(raw_request).release_session(body.session_id)
    return JSONResponse(content={"released_blocks": int(released)})


def attach_router(app: FastAPI):
    app.include_router(router)
