"""
The recovery playbook: belief -> action. Deterministic given a belief.

This lives on its own because it SHIPS. It used to sit in `teacher.py`, which meant the
on-device student imported, at runtime, the one module in the project that exists only to
be handed the ground-truth answer. Nothing that ships may depend on that module.
"""
from __future__ import annotations

PLAYBOOK = {
    "spot":        ("hop_channel", "a clean channel exists; move to it"),
    "sweep":       ("hop_channel", "hop ahead of the sweep"),
    "barrage":     ("fallback_to_lora", "all channels hot; trade rate for a link"),
    "reactive":    ("change_tdma_slot", "deny the jammer its trigger; do NOT hop"),
    "fading":      ("set_tx_power", "geometry, not an attacker; raise margin, NEVER hop"),
    "node_loss":   ("reroute", "the peer is gone; route around it"),
    "congestion":  ("change_tdma_slot", "back off and separate transmitters"),
    "hidden_term": ("change_tdma_slot", "separate the colliding transmitters"),
}
