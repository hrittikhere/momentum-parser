import json
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import errors as genai_errors
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

API_BASE = "https://api.momentum.io/v1/meetings"
MAX_PAGE_SIZE = 50
MAX_REQUESTS = 100
KEYWORDS = ["neocloud", "gpu", "gpus", "nvidia", "cuda", "graphics card", "compute"]


def fetch_all_meetings(api_key, from_date=None, to_date=None):
    """Fetch meetings from Momentum API, respecting the 100-request rate limit.

    Strategy: use pageSize=50 to maximize data per request.
    First request discovers pageCount, then fetch remaining pages up to 100 total requests.
    """
    headers = {"X-API-Key": api_key}
    params = {"pageSize": MAX_PAGE_SIZE, "pageNumber": 1}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    all_meetings = []
    requests_made = 0

    # First request to discover total pages
    resp = requests.get(API_BASE, headers=headers, params=params, timeout=30)
    requests_made += 1

    if resp.status_code == 401:
        return None, "Invalid API key. Please check and try again."
    if resp.status_code == 429:
        return None, "Rate limit exceeded. Please wait 15 minutes and try again."
    if resp.status_code != 200:
        return None, f"API error {resp.status_code}: {resp.text}"

    data = resp.json()
    all_meetings.extend(data.get("meetings", []))
    total_pages = data.get("pageCount", 1)

    # Fetch remaining pages up to our request budget
    max_page = min(total_pages, MAX_REQUESTS)

    for page in range(2, max_page + 1):
        if requests_made >= MAX_REQUESTS:
            break

        params["pageNumber"] = page
        resp = requests.get(API_BASE, headers=headers, params=params, timeout=30)
        requests_made += 1

        if resp.status_code == 429:
            # Hit rate limit — return what we have so far
            break
        if resp.status_code != 200:
            continue

        page_data = resp.json()
        meetings = page_data.get("meetings", [])
        if not meetings:
            break
        all_meetings.extend(meetings)

    return all_meetings, f"Fetched {len(all_meetings)} meetings using {requests_made}/{MAX_REQUESTS} API requests (out of {total_pages} total pages)."


def search_transcript(transcript, keywords):
    """Search transcript entries for keyword matches. Returns matching snippets."""
    if not transcript or not transcript.get("entries"):
        return []

    matches = []
    entries = transcript["entries"]
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        if pattern.search(text):
            # Grab surrounding context (2 entries before/after)
            start = max(0, i - 2)
            end = min(len(entries), i + 3)
            context_entries = entries[start:end]
            snippet = "\n".join(
                f"[{e.get('speaker', {}).get('name', 'Unknown')}]: {e.get('text', '')}"
                for e in context_entries
            )
            matches.append({
                "text": text,
                "speaker": entry.get("speaker", {}).get("name", "Unknown"),
                "timestamp": entry.get("timestamp", ""),
                "context": snippet,
            })

    return matches


def filter_meetings_by_keywords(meetings, keywords):
    """Filter meetings that mention keywords in title or transcript."""
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    results = []

    for meeting in meetings:
        title = meeting.get("title", "")
        title_match = bool(pattern.search(title))
        transcript_matches = search_transcript(meeting.get("transcript"), keywords)

        if title_match or transcript_matches:
            attendees = meeting.get("attendees", [])
            external = [a for a in attendees if not a.get("isInternal", True)]
            internal = [a for a in attendees if a.get("isInternal", False)]

            results.append({
                "id": meeting.get("id"),
                "title": title,
                "startTime": meeting.get("startTime"),
                "endTime": meeting.get("endTime"),
                "host": meeting.get("host", {}),
                "externalAttendees": [
                    {"name": a.get("name", ""), "email": a.get("email", "")}
                    for a in external
                ],
                "internalAttendees": [
                    {"name": a.get("name", ""), "email": a.get("email", "")}
                    for a in internal
                ],
                "titleMatch": title_match,
                "transcriptMatches": transcript_matches,
                "matchCount": len(transcript_matches) + (1 if title_match else 0),
                "fullTranscript": _format_transcript(meeting.get("transcript")),
            })

    results.sort(key=lambda x: x["matchCount"], reverse=True)
    return results


def _format_transcript(transcript):
    if not transcript or not transcript.get("entries"):
        return ""
    return "\n".join(
        f"[{e.get('speaker', {}).get('name', 'Unknown')}]: {e.get('text', '')}"
        for e in transcript["entries"]
    )


def analyze_patterns(filtered_meetings):
    """Extract patterns from filtered meetings for content ideation."""
    if not filtered_meetings:
        return {}

    all_keyword_mentions = Counter()
    speakers = Counter()
    topics_by_meeting = []
    external_companies = Counter()
    timeline = Counter()
    all_snippets = []

    keyword_pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in KEYWORDS) + r")\b", re.IGNORECASE
    )

    for m in filtered_meetings:
        # Count keyword frequency
        for match in m.get("transcriptMatches", []):
            for kw_match in keyword_pattern.findall(match["text"]):
                all_keyword_mentions[kw_match.lower()] += 1
            speakers[match["speaker"]] += 1
            all_snippets.append(match["text"])

        # Track external attendees (potential customer segments)
        for att in m.get("externalAttendees", []):
            email = att.get("email", "")
            if email:
                domain = email.split("@")[-1] if "@" in email else ""
                if domain:
                    external_companies[domain] += 1

        # Timeline
        start = m.get("startTime", "")
        if start:
            day = start[:10]
            timeline[day] += 1

        # Collect topic hints from titles
        topics_by_meeting.append(m.get("title", ""))

    # Detect content gaps and confusion signals
    content_gaps = detect_content_gaps(filtered_meetings)

    return {
        "totalMatches": len(filtered_meetings),
        "keywordFrequency": dict(all_keyword_mentions.most_common(20)),
        "topSpeakers": dict(speakers.most_common(10)),
        "externalCompanies": dict(external_companies.most_common(15)),
        "meetingsPerDay": dict(sorted(timeline.items())),
        "meetingTitles": topics_by_meeting,
        "sampleSnippets": all_snippets[:50],
        "contentGaps": content_gaps,
    }


CONFUSION_SIGNALS = re.compile(
    r"(i don'?t understand|confused|what do you mean|can you explain|"
    r"how do i|how does that work|what'?s the difference|"
    r"i'?m not sure|unclear|doesn'?t make sense|"
    r"where do i find|where is the|how to|"
    r"is there documentation|is there a guide|"
    r"can you walk me through|can you show me|"
    r"i thought it was|wait,? so|that'?s confusing)",
    re.IGNORECASE,
)

CONTENT_REQUEST_SIGNALS = re.compile(
    r"(do you have.*(doc|guide|tutorial|article|video|walkthrough|example)|"
    r"is there a.*(doc|guide|tutorial|blog|resource|page)|"
    r"where can i.*(read|learn|find|see)|"
    r"any.*(documentation|resources|examples|guides)|"
    r"send me.*(link|doc|info|resource)|"
    r"i wish there was|it would be nice to have|"
    r"we need.*(doc|content|guide|training)|"
    r"FAQ|knowledge base|help center|support article)",
    re.IGNORECASE,
)

PAIN_POINT_SIGNALS = re.compile(
    r"(frustrat|annoying|difficult|struggle|pain point|"
    r"takes too long|too complicated|hard to|"
    r"doesn'?t work|broken|bug|issue|problem with|"
    r"we keep running into|keeps happening|"
    r"workaround|hack|manually)",
    re.IGNORECASE,
)

FEATURE_REQUEST_SIGNALS = re.compile(
    r"(would be great if|it would be nice|can you add|"
    r"feature request|we need|we want|we'?d love|"
    r"any plans to|on the roadmap|"
    r"could you support|ability to|"
    r"wish we could|if only)",
    re.IGNORECASE,
)


def detect_content_gaps(filtered_meetings):
    """Scan transcripts for signals of confusion, content requests, pain points, and feature requests."""
    categories = {
        "confusion": {"pattern": CONFUSION_SIGNALS, "items": []},
        "contentRequests": {"pattern": CONTENT_REQUEST_SIGNALS, "items": []},
        "painPoints": {"pattern": PAIN_POINT_SIGNALS, "items": []},
        "featureRequests": {"pattern": FEATURE_REQUEST_SIGNALS, "items": []},
    }

    for m in filtered_meetings:
        transcript = m.get("fullTranscript", "")
        if not transcript:
            continue

        lines = transcript.split("\n")
        for i, line in enumerate(lines):
            for cat_name, cat in categories.items():
                match = cat["pattern"].search(line)
                if match:
                    # Get surrounding context
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])

                    cat["items"].append({
                        "meeting": m.get("title", "Untitled"),
                        "meetingDate": m.get("startTime", ""),
                        "matchedPhrase": match.group(0),
                        "line": line.strip(),
                        "context": context,
                    })

    # Deduplicate and limit
    for cat_name in categories:
        seen = set()
        deduped = []
        for item in categories[cat_name]["items"]:
            key = item["line"][:80]
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        categories[cat_name]["items"] = deduped[:30]

    return {
        "confusion": categories["confusion"]["items"],
        "contentRequests": categories["contentRequests"]["items"],
        "painPoints": categories["painPoints"]["items"],
        "featureRequests": categories["featureRequests"]["items"],
        "summary": {
            "confusionCount": len(categories["confusion"]["items"]),
            "contentRequestCount": len(categories["contentRequests"]["items"]),
            "painPointCount": len(categories["painPoints"]["items"]),
            "featureRequestCount": len(categories["featureRequests"]["items"]),
        },
    }


def cluster_signals_with_claude(content_gaps, claude_api_key):
    """Send all detected signals to Claude to cluster into recurring themes with root-cause reasoning."""
    signals = []
    category_map = {
        "confusion": content_gaps.get("confusion", []),
        "pain_point": content_gaps.get("painPoints", []),
        "content_request": content_gaps.get("contentRequests", []),
        "feature_request": content_gaps.get("featureRequests", []),
    }
    for category, items in category_map.items():
        for item in items:
            signals.append({
                "category": category,
                "quote": item.get("line", ""),
                "context": item.get("context", ""),
                "meeting": item.get("meeting", ""),
                "date": item.get("meetingDate", "")[:10] if item.get("meetingDate") else "",
            })

    if not signals:
        return []

    prompt = f"""You are analyzing customer call signals to find recurring underlying problems.

Below are {len(signals)} signals detected across customer sales and support calls. Each has a category, the exact quote, surrounding context, meeting name, and date.

Group them into 5–10 distinct themes. Each theme should represent a single underlying problem — not a symptom. Prioritize themes that appear across the most distinct calls.

For each theme return:
- "theme": concise name of the underlying problem (not the symptom)
- "root_cause": 2–4 sentences explaining WHY customers keep hitting this — what mental model is wrong, what expectation is unmet, what knowledge gap exists. Be specific and analytical.
- "signal_types": array of signal categories present (confusion, pain_point, content_request, feature_request)
- "call_count": number of distinct calls this theme appears in
- "signal_count": total individual signal hits across all calls
- "quotes": up to 3 representative verbatim quotes, each with "text", "meeting", and "date"

Rank themes by call_count descending.

Return ONLY valid JSON in this format:
{{"themes": [...]}}

SIGNALS:
{json.dumps(signals, indent=2)}"""

    try:
        client = genai.Client(api_key=claude_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        return data.get("themes", [])
    except genai_errors.ClientError as e:
        if "API_KEY_INVALID" in str(e) or "401" in str(e):
            return {"error": "Invalid Gemini API key."}
        return {"error": f"Clustering failed: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Clustering failed: {str(e)}"}


def format_meetings_as_markdown(meetings, from_date, to_date, generated_at):
    week_start = from_date[:10]
    week_end = to_date[:10]
    lines = [
        f"# Weekly Call Report — {week_start} to {week_end}",
        f"_Generated: {generated_at}_  ",
        f"_Total calls: {len(meetings)}_",
        "",
        "---",
        "",
    ]

    for i, m in enumerate(meetings):
        title = m.get("title") or "Untitled Meeting"
        start = m.get("startTime", "")
        end = m.get("endTime", "")
        host = m.get("host") or {}
        attendees = m.get("attendees") or []
        external = [a for a in attendees if not a.get("isInternal", True)]
        internal = [a for a in attendees if a.get("isInternal", False)]

        lines.append(f"## {i + 1}. {title}")
        lines.append(f"**Date:** {start[:10] if start else 'Unknown'}  ")
        if start:
            lines.append(f"**Start:** {start}  ")
        if end:
            lines.append(f"**End:** {end}  ")

        host_str = host.get("name", "Unknown")
        if host.get("email"):
            host_str += f" ({host['email']})"
        lines.append(f"**Host:** {host_str}  ")

        if external:
            ext_str = ", ".join(
                f"{a.get('name', '')} <{a.get('email', '')}>" for a in external if a.get("name") or a.get("email")
            )
            if ext_str:
                lines.append(f"**External:** {ext_str}  ")

        if internal:
            int_str = ", ".join(
                f"{a.get('name', '')} <{a.get('email', '')}>" for a in internal if a.get("name") or a.get("email")
            )
            if int_str:
                lines.append(f"**Internal:** {int_str}  ")

        lines.append("")

        transcript = m.get("transcript")
        if transcript and transcript.get("entries"):
            lines.append("### Transcript")
            lines.append("")
            for entry in transcript["entries"]:
                speaker = entry.get("speaker", {}).get("name", "Unknown")
                text = entry.get("text", "")
                ts = entry.get("timestamp", "")
                prefix = f"**[{speaker}]**" + (f" _{ts}_" if ts else "")
                lines.append(f"{prefix}: {text}")
        else:
            lines.append("_No transcript available._")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


@app.route("/api/weekly-report", methods=["POST"])
def weekly_report():
    body = request.json or {}
    api_key = body.get("apiKey", "").strip()

    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    now = datetime.now(timezone.utc)
    to_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    from_date = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

    meetings, status_msg = fetch_all_meetings(api_key, from_date, to_date)
    if meetings is None:
        return jsonify({"error": status_msg}), 400

    generated_at = now.strftime("%Y-%m-%d %H:%M UTC")
    markdown = format_meetings_as_markdown(meetings, from_date, to_date, generated_at)

    return jsonify({
        "markdown": markdown,
        "count": len(meetings),
        "status": status_msg,
        "filename": f"weekly-calls-{from_date[:10]}-to-{to_date[:10]}.md",
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fetch", methods=["POST"])
def fetch_meetings():
    body = request.json or {}
    api_key = body.get("apiKey", "").strip()
    claude_api_key = body.get("claudeApiKey", "").strip()
    mode = body.get("mode", "keywords")  # "keywords" or "last5days"
    custom_keywords = body.get("keywords", "")

    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    keywords = KEYWORDS
    if custom_keywords:
        keywords = [k.strip() for k in custom_keywords.split(",") if k.strip()]

    now = datetime.now(timezone.utc)
    to_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if mode == "last5days":
        from_date = (now - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif mode == "lastweek":
        from_date = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        from_date = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    meetings, status_msg = fetch_all_meetings(api_key, from_date, to_date)
    if meetings is None:
        return jsonify({"error": status_msg}), 400

    filtered = filter_meetings_by_keywords(meetings, keywords)
    patterns = analyze_patterns(filtered)

    response = {
        "status": status_msg,
        "totalFetched": len(meetings),
        "totalMatching": len(filtered),
        "meetings": filtered,
        "patterns": patterns,
        "themes": None,
        "themesError": None,
    }

    if claude_api_key:
        themes = cluster_signals_with_claude(patterns.get("contentGaps", {}), claude_api_key)
        if isinstance(themes, dict) and "error" in themes:
            response["themesError"] = themes["error"]
        else:
            response["themes"] = themes

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
