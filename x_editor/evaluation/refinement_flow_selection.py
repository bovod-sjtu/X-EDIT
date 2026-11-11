"""Exact one-artifact flow selection over observed attempts.

Each candidate is a series network: delivery, dimension capacities, and each
preservation constraint. Its bottleneck is both a feasible flow and a minimum
cut certificate. A binary choice permits one complete artifact per task; it
does not add incompatible attempts as parallel fractional deliveries. Exhaustive
enumeration solves this finite constrained problem exactly. Capacities remain
evaluator-derived proxies, not theoretical module ceilings or acoustic truth.
"""

from __future__ import annotations

import math


def retention_constraints(utility):
    """Preserve known baseline constraints and already-satisfied regional goals."""
    return {**utility["preservation"], **{
        key: grade for key, grade in utility.get("critical_constraints", {}).items()
        if grade is not None and grade >= 2
    }}


def observed_constraints(utility):
    return {**utility["preservation"], **utility.get("critical_constraints", {})}


def candidate_flow(attempt):
    utility = attempt["utility"]
    capacities = {"delivery": float(attempt["delivered"])}
    if utility.get('design_pending'):
        capacities['unmeasured_quality'] = 0.0
    capacities.update({k: v[0] for k, v in utility["dimensions"].items()})
    capacities.update({f"preserve:{k}": (v or 0) / 3 for k, v in utility["preservation"].items()})
    capacities.update({f"required:{k}": (v or 0) / 3
                       for k, v in utility.get("critical_constraints", {}).items()})
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in capacities.values()):
        raise ValueError("flow capacity must be finite and in [0, 1]")
    cut = min(capacities, key=capacities.get)
    value = capacities[cut]
    if not math.isclose(value, utility["lower"], abs_tol=1e-12):
        raise ValueError("candidate flow differs from its recorded utility")
    edges = [
        {
            "from": f"v{i}",
            "to": f"v{i + 1}",
            "constraint": name,
            "capacity": capacity,
            "flow": value,
        }
        for i, (name, capacity) in enumerate(capacities.items())
    ]
    return {
        "value": value,
        "edges": edges,
        "minimum_cut_constraint": cut,
        "minimum_cut_capacity": value,
    }


def select_flow_attempt(attempts):
    indices = [a["attempt_index"] for a in attempts]
    if len(indices) != len(set(indices)):
        raise ValueError("attempt indices must be unique")
    ordered = sorted(attempts, key=lambda a: a["attempt_index"])
    anchor = next((a for a in ordered if a["delivered"]), None)
    chosen, decisions = None, []
    for attempt in ordered:
        losses = []
        if anchor is not None:
            losses = [
                k
                for k, v in retention_constraints(anchor["utility"]).items()
                if v is not None
                and (
                    observed_constraints(attempt["utility"]).get(k) is None
                    or observed_constraints(attempt["utility"])[k] < v
                )
            ]
        eligible = attempt["delivered"] and not losses
        network = candidate_flow(attempt)
        decisions.append(
            {
                "attempt_index": attempt["attempt_index"],
                "eligible": eligible,
                "preservation_losses_or_unknowns": losses,
                "network": network,
            }
        )
        if eligible and (chosen is None or network["value"] > chosen["utility"]["lower"]):
            chosen = attempt
    for decision in decisions:
        decision["selected"] = chosen is not None and (
            decision["attempt_index"] == chosen["attempt_index"]
        )
    return chosen, decisions
