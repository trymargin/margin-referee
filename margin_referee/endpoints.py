"""A model endpoint for the replay: OpenRouter, or a model in your OWN account (Bedrock, Vertex AI,
Azure OpenAI, any OpenAI-compatible endpoint). Ported from Margin's hosted product.

⛔ THE CREDENTIAL IS NEVER STORED OR LOGGED. It lives in this object for the replay and is gone
with it; ``__repr__`` hides it, and an HTTP error carries the endpoint's status and body, never
the request headers.
⛔ THE PRICE IS THE CUSTOMER'S, AND IT IS REQUIRED. Their own account is billed at their own rate
(including any negotiated discount), which Margin cannot know. An unpriced candidate would read as
free and win on cost, so ``parse`` refuses one rather than guessing.
⛔ A CUSTOM URL IS HTTPS AND PUBLIC. Margin's servers make the call, so a URL that resolves to a
private, loopback or link-local address is refused (no reaching into our own network).
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

KINDS = ("openrouter", "openai_compatible", "azure_openai", "vertex_ai", "bedrock")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
_NAME = re.compile(r"^[A-Za-z0-9._:/@-]{1,200}$")
_SLUG = re.compile(r"^[a-z0-9-]{1,63}$")
TIMEOUT_S = 120


class EndpointError(ValueError):
    """The endpoint description cannot be used; the message says why."""


@dataclass(frozen=True)
class Endpoint:
    kind: str
    url: str
    model: str
    input_per_mtok: float
    output_per_mtok: float
    headers: dict = field(repr=False)

    @property
    def label(self) -> str:
        """What the verdict and the best-of-N count call this candidate."""
        return f"{self.kind}:{self.model}"

    def __repr__(self) -> str:  # never the credential
        return f"Endpoint({self.label!r}, {self.url!r})"


def _need(d: dict, key: str, pattern: re.Pattern = _NAME) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not pattern.match(v.strip()):
        raise EndpointError(f"candidate_endpoint.{key} is missing or not a plain name")
    return v.strip()


def _secret(d: dict, key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip() or len(v) > 8192:
        raise EndpointError(f"candidate_endpoint.{key} is required (it is used for this replay "
                            "only and never stored)")
    return v.strip()


def _public_https(url: str) -> str:
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or not u.hostname:
        raise EndpointError("candidate_endpoint.base_url must be an https URL")
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(u.hostname, u.port or 443)}
    except OSError:
        raise EndpointError(f"candidate_endpoint.base_url: {u.hostname} does not resolve")
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved \
                or ip.is_multicast or ip.is_unspecified:
            raise EndpointError("candidate_endpoint.base_url resolves to a private address, "
                                "which Margin will not call")
    return url.rstrip("/")


def parse(d: object) -> Endpoint:
    """An ``Endpoint`` from the tool's ``candidate_endpoint`` object, or ``EndpointError``."""
    if not isinstance(d, dict):
        raise EndpointError("candidate_endpoint is an object")
    kind = d.get("kind")
    if kind not in KINDS:
        raise EndpointError(f"candidate_endpoint.kind is one of {', '.join(KINDS)}")
    if kind == "openrouter":  # billed cost comes back on every answer, so no price is needed
        return Endpoint(kind, OPENROUTER, _need(d, "model"), -1.0, -1.0,
                        {"Authorization": "Bearer " + _secret(d, "api_key")})
    price = d.get("price") or {}
    try:
        pin, pout = float(price["input_per_mtok"]), float(price["output_per_mtok"])
    except (KeyError, TypeError, ValueError):
        raise EndpointError("candidate_endpoint.price needs input_per_mtok and output_per_mtok "
                            "(USD per million tokens, your own rate): an unpriced candidate "
                            "would read as free")
    if not (0 <= pin < 1000 and 0 <= pout < 1000):
        raise EndpointError("candidate_endpoint.price is USD per million tokens, 0 to 1000")
    if kind == "openai_compatible":
        raw = d.get("base_url")
        url = _public_https(raw.strip() if isinstance(raw, str) and len(raw) < 300 else "")
        url, model = url + "/chat/completions", _need(d, "model")
        headers = {"Authorization": "Bearer " + _secret(d, "api_key")}
    elif kind == "azure_openai":
        res = _need(d, "resource", _SLUG)
        model = _need(d, "deployment")
        ver = _need(d, "api_version") if d.get("api_version") else "2024-10-21"
        url = (f"https://{res}.openai.azure.com/openai/deployments/"
               f"{urllib.parse.quote(model, safe='')}/chat/completions?api-version={ver}")
        headers = {"api-key": _secret(d, "api_key")}
    elif kind == "vertex_ai":
        project, region, model = _need(d, "project", _SLUG), _need(d, "region", _SLUG), \
            _need(d, "model")
        url = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/"
               f"{region}/endpoints/openapi/chat/completions")
        headers = {"Authorization": "Bearer " + _secret(d, "access_token")}
    else:  # bedrock: the Converse API with a Bedrock API key (no request signing needed)
        region, model = _need(d, "region", _SLUG), _need(d, "model")
        url = (f"https://bedrock-runtime.{region}.amazonaws.com/model/"
               f"{urllib.parse.quote(model, safe='')}/converse")
        headers = {"Authorization": "Bearer " + _secret(d, "api_key")}
    return Endpoint(kind, url, model, pin, pout, headers)


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # the endpoint's answer, never our request headers
        raise RuntimeError(f"HTTP {e.code} from the endpoint: {e.read()[:300]!r}") from None


def _bedrock_body(messages: list, max_tokens: int, temperature: float) -> dict:
    system = [{"text": m["content"]} for m in messages if m.get("role") == "system"]
    msgs = [{"role": m["role"], "content": [{"text": str(m.get("content") or "")}]}
            for m in messages if m.get("role") in ("user", "assistant")]
    config = {"maxTokens": max_tokens, "temperature": temperature}
    body = {"messages": msgs, "inferenceConfig": config}
    return {**body, "system": system} if system else body


def chat_for(ep: Endpoint):
    """A ``chat`` callable shaped like ``providers.openrouter.chat`` for the replay. The OpenRouter
    routing policy kwargs are accepted and ignored: the call goes to the customer's own account."""
    def chat(messages, *, model=None, temperature=0.0, max_tokens=1024, response_format=None,
             tools=None, tool_choice=None, **_policy):
        if ep.kind == "bedrock":
            if tools:
                # ponytail: Converse takes tools in its own shape (toolConfig / toolUse), not
                # mapped here; the case is left unanswered and named, never run without them.
                raise RuntimeError("tool definitions are not supported on a bedrock endpoint "
                                   "yet; use an OpenAI-compatible endpoint for a tool case")
            calls = None
            d = _post(ep.url, ep.headers, _bedrock_body(messages, max_tokens, temperature))
            text = "".join(c.get("text", "") for c in
                           ((d.get("output") or {}).get("message") or {}).get("content") or [])
            u = d.get("usage") or {}
            tin, tout = int(u.get("inputTokens") or 0), int(u.get("outputTokens") or 0)
            finish = "length" if d.get("stopReason") == "max_tokens" else "stop"
        else:
            body = {"model": ep.model, "messages": messages, "temperature": temperature,
                    "max_tokens": max_tokens}
            if response_format:
                body["response_format"] = response_format
            if tools:
                body["tools"] = tools
                if tool_choice is not None:
                    body["tool_choice"] = tool_choice
            if ep.kind == "openrouter":
                body["usage"] = {"include": True}
            d = _post(ep.url, ep.headers, body)
            choice = (d.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content")) or ""
            calls = (choice.get("message") or {}).get("tool_calls") or []
            u = d.get("usage") or {}
            tin, tout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            finish = choice.get("finish_reason") or "stop"
        if not (tin or tout):
            raise RuntimeError("the endpoint reported no token usage, so the call cannot be "
                               "priced")
        if ep.kind == "openrouter":
            if u.get("cost") is None:
                raise RuntimeError("OpenRouter returned no billed cost for this call")
            cost = float(u["cost"])
        else:
            cost = (tin * ep.input_per_mtok + tout * ep.output_per_mtok) / 1e6
        return SimpleNamespace(text=text, cost_usd=cost, finish_reason=finish, input_tokens=tin,
                               output_tokens=tout, cache_read_tokens=0, model=ep.label,
                               tool_calls=calls)
    return chat


def describe(ep: Optional[Endpoint]) -> Optional[dict]:
    """What a verdict may say about the endpoint (never the credential)."""
    if ep is None:
        return None
    if ep.kind == "openrouter":
        return {"kind": ep.kind, "model": ep.model, "price_source": "OpenRouter's billed cost"}
    return {"kind": ep.kind, "model": ep.model, "host": urllib.parse.urlparse(ep.url).hostname,
            "price_per_mtok": {"input": ep.input_per_mtok, "output": ep.output_per_mtok},
            "price_source": "the rate you supplied for your own account"}
