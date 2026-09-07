"""
FreeToken OpenAI / Anthropic Compatible HTTP Serving Server
Reference: arXiv:2608.16157
"""

from __future__ import annotations
import time
import json
import uuid
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from freetoken.engine import FreeTokenEngine, EngineConfig, MoEModelConfig


# API Request/Response Schemas (OpenAI Compatible)
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "freetoken-moe-default"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


def create_app(engine: Optional[FreeTokenEngine] = None) -> FastAPI:
    app = FastAPI(title="FreeToken Inference Server", version="0.1.0")

    if engine is None:
        # Default lightweight configuration for local edge testing
        m_cfg = MoEModelConfig(
            vocab_size=1000,
            hidden_dim=256,
            intermediate_dim=512,
            num_layers=4,
            num_experts=8,
            top_k=2
        )
        e_cfg = EngineConfig(
            total_vram_gb=4.0,
            base_model_vram_gb=1.0,
            reserved_vram_gb=0.5,
            initial_expert_capacity=4
        )
        engine = FreeTokenEngine(m_cfg, e_cfg)

    app.state.engine = engine

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": "FreeToken (arXiv:2608.16157)"}

    @app.get("/v1/metrics")
    def metrics():
        return app.state.engine.get_metrics()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        engine: FreeTokenEngine = app.state.engine

        # Mock tokenization for demonstration / testing
        # Convert prompt text characters to mock integer tokens
        combined_text = " ".join([m.content for m in req.messages])
        prompt_ids = [ord(c) % engine.model_cfg.vocab_size for c in combined_text]
        if not prompt_ids:
            prompt_ids = [1]

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        if req.stream:
            async def event_generator():
                for token in engine.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=req.max_tokens or 64,
                    temperature=req.temperature or 0.7,
                    top_p=req.top_p or 0.9
                ):
                    chunk_data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chr(token % 95 + 32)},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming
        generated_tokens = list(engine.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=req.max_tokens or 64,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9
        ))
        generated_text = "".join([chr(t % 95 + 32) for t in generated_tokens])

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=req.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ]
        )

    return app
