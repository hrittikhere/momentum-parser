import json
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

from app import (
    KEYWORDS,
    analyze_patterns,
    fetch_all_meetings,
    filter_meetings_by_keywords,
    filter_meetings_by_person,
)

mcp = FastMCP("Momentum Parser")


def _serialize_call(m):
    host = m.get("host") or {}
    attendees = m.get("attendees") or []
    entries = (m.get("transcript") or {}).get("entries") or []
    return {
        "title": m.get("title") or "Untitled",
        "date": (m.get("startTime") or "")[:10],
        "start_time": m.get("startTime"),
        "end_time": m.get("endTime"),
        "host": host.get("name"),
        "external_attendees": [
            {"name": a.get("name"), "email": a.get("email")}
            for a in attendees
            if not a.get("isInternal", True)
        ],
        "internal_attendees": [
            {"name": a.get("name"), "email": a.get("email")}
            for a in attendees
            if a.get("isInternal", False)
        ],
        "transcript": "\n".join(
            f"[{e.get('speaker', {}).get('name', 'Unknown')}]: {e.get('text', '')}"
            for e in entries
        ),
    }


@mcp.tool()
def get_person_calls(momentum_api_key: str, person_name: str, days: int = 7) -> str:
    """
    Fetch all Momentum calls featuring a specific person.

    Returns each call with title, date, attendees, the person's own statements
    extracted, and the full transcript. Use this to capture what someone was
    hearing and saying on calls — ideal for drafting LinkedIn posts or building
    a Notion page of their customer insights.

    Args:
        momentum_api_key: Momentum API key.
        person_name: Full or partial name to match (e.g. "Lukas Gentele").
        days: How many days back to search (default 7).
    """
    days = int(days)
    now = datetime.now(timezone.utc)
    to_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    meetings, status_msg = fetch_all_meetings(momentum_api_key, from_date, to_date)
    if meetings is None:
        return json.dumps({"error": status_msg})

    filtered = filter_meetings_by_person(meetings, person_name)

    calls = []
    for m in filtered:
        base = _serialize_call(m)
        host = m.get("host") or {}
        entries = (m.get("transcript") or {}).get("entries") or []
        base["is_host"] = person_name.lower() in (host.get("name") or "").lower()
        base["person_statements"] = [
            e.get("text", "")
            for e in entries
            if person_name.lower() in (e.get("speaker", {}).get("name") or "").lower()
        ]
        calls.append(base)

    return json.dumps(
        {
            "person": person_name,
            "period": f"{from_date[:10]} to {to_date[:10]}",
            "calls_found": len(filtered),
            "total_calls_searched": len(meetings),
            "calls": calls,
        },
        indent=2,
    )


@mcp.tool()
def get_weekly_calls(momentum_api_key: str, days: int = 7) -> str:
    """
    Fetch all Momentum calls from the past N days — no keyword filtering.

    Returns every call with title, date, host, attendees, and full transcript.
    Use this to build a weekly call summary page in Notion or to give an AI
    a complete picture of what happened across all customer conversations.

    Args:
        momentum_api_key: Momentum API key.
        days: How many days back to fetch (default 7).
    """
    days = int(days)
    now = datetime.now(timezone.utc)
    to_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    meetings, status_msg = fetch_all_meetings(momentum_api_key, from_date, to_date)
    if meetings is None:
        return json.dumps({"error": status_msg})

    return json.dumps(
        {
            "period": f"{from_date[:10]} to {to_date[:10]}",
            "total_calls": len(meetings),
            "calls": [_serialize_call(m) for m in meetings],
            "api_status": status_msg,
        },
        indent=2,
    )


@mcp.tool()
def analyze_keyword_calls(
    momentum_api_key: str, keywords: str = "", days: int = 30
) -> str:
    """
    Fetch and analyze Momentum calls matching specific keywords.

    Returns keyword frequency, top speakers, external companies, content gaps
    (confusion signals, pain points, feature requests), and matched calls with
    transcript snippets. Use this to populate a Notion insights database or
    generate a customer intelligence summary.

    Args:
        momentum_api_key: Momentum API key.
        keywords: Comma-separated keywords (defaults to GPU/AI cloud topics).
        days: How many days back to search (default 30).
    """
    days = int(days)
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]

    now = datetime.now(timezone.utc)
    to_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    meetings, status_msg = fetch_all_meetings(momentum_api_key, from_date, to_date)
    if meetings is None:
        return json.dumps({"error": status_msg})

    filtered = filter_meetings_by_keywords(meetings, kw_list) if kw_list else meetings
    patterns = analyze_patterns(filtered)

    return json.dumps(
        {
            "period": f"{from_date[:10]} to {to_date[:10]}",
            "keywords_searched": kw_list,
            "total_calls_fetched": len(meetings),
            "matching_calls": len(filtered),
            "patterns": patterns,
            "api_status": status_msg,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
