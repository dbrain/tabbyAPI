"""POST /v1/systemone: typed decision questions answered from one label token's logits.

Implements the System One contract shared with koblem's emulation (llama.cpp's TypeSafe-compatible
/v1/systemone wire API). Each question gets its own prompt, all of them starting with the same state so
that the questions after the first resume from the first one's cached state. The model reads the prompt
and the logits of the position after it are read for the option labels only:

    p = softmax(logit[label] / T)        over the labels, T = model.systemone_temperature
    confidence = 1 - H(p) / ln(n)

The prompt text below is byte-exact with the contract: koblem's emulation builds the same string and
reads the same position through /v1/completions, and the two must agree.
"""

import asyncio
import json
import math
import os
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from common import model
from common.logger import xlogger
from common.networking import run_with_request_disconnect
from common.tabby_config import config

SYSTEM_PROMPT = (
    "You are a decision model. Read the state and answer the question with the label of one option only."
)
CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]
MAX_SCORE_LEVELS = 10


class ODict(dict):
    """A parsed JSON object that remembers which of its keys were repeated (json keeps the last)."""

    dups: set = frozenset()


def _pairs(pairs):
    d = ODict()
    dups = set()
    for k, v in pairs:
        if k in d:
            dups.add(k)
        d[k] = v
    d.dups = dups
    return d


def loads(raw: bytes):
    return json.loads(raw, object_pairs_hook=_pairs)


class SystemOneError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def error_response(status: int, message: str) -> JSONResponse:
    kind = {400: "invalid_request_error", 501: "not_supported_error", 503: "unavailable_error"}.get(
        status, "server_error"
    )
    return JSONResponse({"error": {"code": status, "message": message, "type": kind}}, status_code=status)


def as_text(value) -> str:
    """A string as is; anything else as compact JSON (key order kept, non-ASCII unescaped)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_question(qid: str, q) -> dict:
    if not isinstance(q, dict):
        raise SystemOneError(400, f"questions.{qid} must be an object")
    qtype = q.get("type")
    if qtype not in ("choice", "score", "noul"):
        raise SystemOneError(400, f"questions.{qid}.type must be one of choice, score, noul")
    if "instructions" not in q or q["instructions"] is None:
        raise SystemOneError(400, f"questions.{qid}.instructions is required")
    instructions = as_text(q["instructions"])
    criteria = q.get("criteria")

    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise SystemOneError(400, f"questions.{qid}.criteria must be an object with at least 2 options")
        if len(criteria) > len(CHOICE_LABELS):
            raise SystemOneError(
                400, f"questions.{qid}.criteria has {len(criteria)} options; at most {len(CHOICE_LABELS)} are supported"
            )
        if getattr(criteria, "dups", None):
            raise SystemOneError(400, f"questions.{qid}.criteria has duplicate options: {sorted(criteria.dups)}")
        options = []
        for label, (name, desc) in zip(CHOICE_LABELS, criteria.items()):
            if desc is not None and not isinstance(desc, str):
                raise SystemOneError(400, f"questions.{qid}.criteria.{name} must be a string or null")
            options.append((label, name, desc))
        keys = list(criteria.keys())

    elif qtype == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise SystemOneError(
                400, f"questions.{qid}.criteria must be an array of 2 to {MAX_SCORE_LEVELS} level descriptions"
            )
        if not all(isinstance(level, str) for level in criteria):
            raise SystemOneError(400, f"questions.{qid}.criteria levels must be strings")
        options = [(str(i), level, None) for i, level in enumerate(criteria)]
        keys = [str(i) for i in range(len(criteria))]

    else:
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
            raise SystemOneError(400, f"questions.{qid}.criteria must be an object with optional true/false descriptions")
        for k, v in criteria.items():
            if v is not None and not isinstance(v, str):
                raise SystemOneError(400, f"questions.{qid}.criteria.{k} must be a string or null")
        options = [("A", "Yes", criteria.get("true")), ("B", "No", criteria.get("false"))]
        keys = ["true", "false"]

    return {"id": qid, "type": qtype, "instructions": instructions, "options": options, "keys": keys,
            "criteria": criteria}


def parse_request(body) -> tuple[str, list[dict]]:
    if not isinstance(body, dict):
        raise SystemOneError(400, "request body must be a JSON object")
    for field in ("files", "images"):
        if body.get(field):
            raise SystemOneError(501, "image input not supported yet")
    if "state" not in body or body["state"] is None:
        raise SystemOneError(400, "state is required")
    state = body["state"]
    if not isinstance(state, (str, dict, list)):
        raise SystemOneError(400, "state must be a string, an object or an array")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SystemOneError(400, "questions must be a non-empty object")
    if getattr(questions, "dups", None):
        raise SystemOneError(400, f"duplicate question ids: {sorted(questions.dups)}")
    return as_text(state), [parse_question(qid, q) for qid, q in questions.items()]


def state_head(state: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nState:\n{state}\n\nQuestion: "
    )


def build_prompt(state: str, q: dict) -> str:
    # An empty description counts as none
    lines = "".join(
        f"{label}) {option}" + (f": {desc}" if desc else "") + "\n"
        for label, option, desc in q["options"]
    )
    return (
        state_head(state)
        + f"{q['instructions']}\nOptions:\n{lines}Answer with the label only.<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def probabilities(logits: list[float], temperature: float) -> list[float]:
    z = [x / temperature for x in logits]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def confidence(p: list[float]) -> float:
    h = -sum(x * math.log(x) for x in p if x > 0)
    return min(1.0, max(0.0, 1.0 - h / math.log(len(p))))


def answer(q: dict, p: list[float]) -> dict:
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[0]}
    if q["type"] == "choice":
        best = max(range(len(p)), key=lambda i: p[i])
        return {
            "type": "choice",
            "choice": q["keys"][best],
            "probabilities": dict(zip(q["keys"], p)),
            "confidence": confidence(p),
        }
    return {
        "type": "score",
        "score": sum(i * x for i, x in enumerate(p)),
        "legend": {str(i): level for i, level in enumerate(q["criteria"])},
        "probabilities": {str(i): x for i, x in enumerate(p)},
        "confidence": confidence(p),
    }


_label_ids: dict = {}


def label_token_ids(container, labels: list[str]) -> list[int]:
    out = []
    for label in labels:
        tid = _label_ids.get(label)
        if tid is None:
            ids = container.tokenizer.encode(label, add_bos=False, encode_special_tokens=False).flatten().tolist()
            if len(ids) != 1:
                raise SystemOneError(500, f"label {label!r} is not a single token for this model")
            tid = _label_ids[label] = ids[0]
        out.append(tid)
    return out


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


async def systemone_request(request: Request) -> JSONResponse:
    t0 = time.perf_counter()
    try:
        body = loads(await request.body())
    except Exception:
        return error_response(400, "request body is not valid JSON")

    try:
        state, questions = parse_request(body)
    except SystemOneError as e:
        return error_response(e.status, e.message)

    container = model.container
    if container is None or not getattr(container, "loaded", False) or container.generator is None:
        return error_response(503, "no model is loaded")

    try:
        temperature = float(getattr(config.model, "systemone_temperature", None) or 1.0)
        if not temperature > 0:
            temperature = 1.0

        prompts = [build_prompt(state, q) for q in questions]
        encode = lambda text: container.tokenizer.encode(  # noqa: E731
            text, add_bos=False, encode_special_tokens=True
        )
        ids = [encode(p) for p in prompts]
        id_lists = [t.flatten().tolist() for t in ids]
        head = encode(state_head(state)).flatten().tolist()
        split_at = min(common_prefix(head, t) for t in id_lists)
        # Test only: checkpoint at the prompt's last page boundary, as a regular prefill does, so a repeated
        # request resumes there and its uncached tail is exactly what the caller padded it to
        if request.headers.get("x-systemone-split") == "prompt" and len(questions) == 1:
            split_at = None

        limit = min(container.max_seq_len, container.cache.max_num_tokens - 256)
        for q, t in zip(questions, id_lists):
            if len(t) > limit:
                raise SystemOneError(
                    400, f"question {q['id']}: prompt is {len(t)} tokens, more than the {limit} this model accepts"
                )
        labels = [label_token_ids(container, [label for label, _, _ in q["options"]]) for q in questions]
    except SystemOneError as e:
        return error_response(e.status, e.message)

    path = (request.headers.get("x-systemone-path") or os.environ.get("EXL3_SYSTEMONE_PATH") or "auto").lower()
    # Test only: x-systemone-batch: 0 runs a group's questions one forward each (exllamav3-patches/16)
    batch = request.headers.get("x-systemone-batch", "1") != "0"
    request_id = f"systemone-{uuid.uuid4().hex[:12]}"
    try:
        # A caller that gives up (a koblem timeout under load) must take its queued questions with it, or
        # they run ahead of every later decision; cancelling the task cancels the jobs
        results, used = await run_with_request_disconnect(
            request,
            asyncio.create_task(container.systemone_score(request_id, ids, labels, split_at, path, batch)),
            "systemone request cancelled by the client",
        )
    except HTTPException:
        raise
    except SystemOneError as e:
        return error_response(e.status, e.message)
    except Exception as e:  # noqa: BLE001
        xlogger.error(f"systemone: scoring failed: {e!r}")
        return error_response(500, f"scoring failed: {e}")

    answers = {}
    for q, r in zip(questions, results):
        answers[q["id"]] = answer(q, probabilities(r["logits"], temperature))

    input_tokens = sum(len(t) for t in id_lists)
    cached = sum(r.get("cached_tokens") or 0 for r in results)
    xlogger.info(
        f"systemone: {len(questions)} question(s) · prompt {input_tokens} tokens, {cached} cached · "
        f"{(time.perf_counter() - t0) * 1000:.0f} ms · {used} path"
    )
    return JSONResponse({
        "model": container.model_dir.name,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    })
