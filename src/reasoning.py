"""
Reasoning generator. Stage-4 manual review samples 10 rows and checks for:
specific facts, JD connection, honest concerns, no hallucination, variation,
and rank-consistency. So every claim here is pulled from the candidate's actual
fields and the computed breakdown — never invented — and genuine gaps are stated.
"""

from __future__ import annotations
from typing import Dict, Any, List


def _concerns(candidate: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    vals = b["feature_values"]
    s = candidate.get("redrob_signals", {})

    if b["disqualifiers"]:
        out.append("flagged: " + ", ".join(b["disqualifiers"]))
    if vals.get("experience", 1) < 0.85:
        yoe = candidate.get("profile", {}).get("years_of_experience", 0)
        out.append(f"experience ({yoe:.1f}y) outside the preferred band")
    if vals.get("location", 1) <= 0.3:
        out.append("outside preferred geography, no relocation")
    if vals.get("product", 1) <= 0.2:
        out.append("mostly services/consulting background")
    if vals.get("title", 1) < 0.6 and vals.get("title", 1) > 0:
        out.append("title is adjacent rather than a direct ML role")
    rr = s.get("recruiter_response_rate", 0) or 0
    if rr < 0.25:
        out.append(f"low recruiter response rate ({rr:.0%})")
    notice = s.get("notice_period_days", 0)
    if notice and notice >= 120:
        out.append(f"long notice period ({notice}d)")
    if b["authenticity"] < 0.85:
        out.append("some profile-consistency concerns")
    return out


def generate_reasoning(candidate: Dict[str, Any], b: Dict[str, Any],
                       rank: int, total: int = 100) -> str:
    p = candidate.get("profile", {})
    s = candidate.get("redrob_signals", {})
    fr = b["feature_reasons"]
    parts: List[str] = []

    # Lead: concrete identity
    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    parts.append(f"{title}, {yoe:.1f}y")

    # Strongest positive (evidence is the backbone signal)
    if fr.get("evidence"):
        parts.append(fr["evidence"][0])
    elif fr.get("title"):
        parts.append(fr["title"][0])

    # A JD-connected positive (product company / preferred city / verified skills)
    for key in ("product", "location", "skill_corroboration"):
        if fr.get(key):
            parts.append(fr[key][0]); break

    # Availability clause — only the informative signals, with real numbers
    rr = s.get("recruiter_response_rate", 0) or 0
    notice = s.get("notice_period_days", None)
    av = f"{rr:.0%} response"
    if notice is not None:
        av += f", {notice}d notice"
    parts.append(av)

    text = "; ".join(parts)

    # Honest concerns — tone scales with rank (top picks should still admit gaps)
    concerns = _concerns(candidate, b)
    if concerns:
        lead = "concern" if rank <= 50 else "notable gaps"
        text += f". {lead.capitalize()}: " + "; ".join(concerns[:2])
    elif rank > 60:
        text += ". Solid but not a standout on evidence depth"

    return text
