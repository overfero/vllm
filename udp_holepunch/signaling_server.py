"""Signaling-only coordinator for UDP hole punching between any number of
peers, each pair getting its own synchronized punch time.

Exchanges endpoint info and a per-pair synchronized punch timestamp over
HTTP. Never touches UDP traffic and never relays anything.
"""
import time
from typing import Dict, FrozenSet

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

app = FastAPI(title="UDP Hole Punch Coordinator")

SYNC_BUFFER_SECONDS = 5.0  # lead time given to both peers before they must start punching

_peers: Dict[str, dict] = {}
_start_at: Dict[FrozenSet[str], float] = {}  # keyed by {peer_id, self_id} - independent per pair


class Registration(BaseModel):
    peer_id: str
    udp_port: int


def _client_ip(request: Request) -> str:
    # zrok (and most HTTP tunnels) terminate in front of this process, so the
    # real client address is in X-Forwarded-For; request.client.host would
    # otherwise just be the local tunnel agent.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


@app.post("/register")
def register(reg: Registration, request: Request) -> dict:
    public_ip = _client_ip(request)  # never trust a client-supplied IP
    prev = _peers.get(reg.peer_id)
    if prev is None or prev["public_ip"] != public_ip or prev["udp_port"] != reg.udp_port:
        # New peer_id, or its endpoint changed (restart/rebind) - drop any frozen
        # sync times involving it so every pair it's part of renegotiates fresh,
        # without disturbing unrelated pairs that don't involve this peer_id.
        for pair in [p for p in _start_at if reg.peer_id in p]:
            del _start_at[pair]
    _peers[reg.peer_id] = {
        "peer_id": reg.peer_id,
        "public_ip": public_ip,
        "udp_port": reg.udp_port,
        "last_seen": time.time(),
    }
    return {"status": "ok", "public_ip": public_ip}


@app.get("/peer/{peer_id}")
def get_peer(peer_id: str, self_id: str = Query(..., description="Caller's own peer_id")) -> dict:
    peer = _peers.get(peer_id)
    if peer is None:
        raise HTTPException(status_code=404, detail="peer not registered yet")
    pair = frozenset((peer_id, self_id))
    if pair not in _start_at:
        _start_at[pair] = time.time() + SYNC_BUFFER_SECONDS
    return {**peer, "start_at": _start_at[pair]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
