"""
prompts.py
Generates technical questions per technology using Google Gemini.
Returns a dictionary: { "questions": { "Python": [...], "Django": [...] } }
"""

import os
import json
import re
import google.generativeai as genai

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# For testing only — will use env var if set, otherwise fall back to hardcoded key
genai.configure(api_key="AIzaSyD0Cl-Y9EShSwg9biO0SrvtJ_28vZ8Lzuo")



DEFAULT_MODEL = "gemini-1.5-flash"


def _extract_text_from_response(resp):
    """
    Try to extract plain text from the Gemini response object.
    The exact structure may vary between SDK versions, so attempt common attributes.
    """
    if resp is None:
        return ""
    # Try common attributes
    text = getattr(resp, "text", None)
    if text:
        return text
    # fallback to str()
    try:
        return str(resp)
    except Exception:
        return ""


def _parse_json_from_text(text: str):
    """
    Try to extract a JSON object substring from the text and parse it.
    """
    # Find first { ... } block (greedy) — helps if model outputs explanations before/after JSON
    match = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Try to load the whole text directly
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_text_by_headings(techs, text):
    """
    Fallback parser: look for headings named after techs (e.g., "Python:", "Django -") and
    collect following lines as questions until the next heading.
    If headings not found, we distribute top N lines among techs as a fallback.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Map of tech -> list
    result = {t: [] for t in techs}

    # Detect headings
    i = 0
    n = len(lines)
    current = None
    while i < n:
        ln = lines[i]
        # check for heading
        matched = None
        for t in techs:
            tl = t.lower()
            if ln.lower().startswith(tl + ":") or ln.lower() == tl or re.match(rf'^{re.escape(t)}\s*[-:]\s*$', ln, flags=re.IGNORECASE):
                matched = t
                break
        if matched:
            current = matched
            i += 1
            # collect until next heading
            while i < n:
                next_ln = lines[i]
                is_next_heading = any(
                    next_ln.lower().startswith(tt.lower() + ":") or next_ln.lower() == tt.lower() for tt in techs
                )
                if is_next_heading:
                    break
                # Remove leading numbering like "1. " or "- "
                cleaned = re.sub(r'^\d+\.\s*', '', next_ln)
                cleaned = re.sub(r'^[\-\•]\s*', '', cleaned)
                if cleaned:
                    result[current].append(cleaned.strip())
                i += 1
        else:
            i += 1

    # If nothing detected (all empty), try distributing lines equally
    if all(len(v) == 0 for v in result.values()):
        # Use lines as questions (remove numbering)
        cleaned_lines = [re.sub(r'^\d+\.\s*', '', ln) for ln in lines]
        # Remove any lines that look like headings or instructions
        filtered = [ln for ln in cleaned_lines if len(ln) > 10]  # avoid tiny lines
        if not filtered:
            return result
        per = max(1, len(filtered) // max(1, len(techs)))
        idx = 0
        for t in techs:
            group = filtered[idx: idx + per]
            result[t].extend(group)
            idx += per
        # if leftover lines, append round-robin
        leftover = filtered[idx:]
        j = 0
        while leftover:
            result[techs[j % len(techs)]].append(leftover.pop(0))
            j += 1

    # Trim each to max 5
    for t in result:
        result[t] = [q for q in result[t] if len(q) > 3][:5]

    return result


def generate_tech_questions(tech_stack: str, min_q: int = 3, max_q: int = 5, model_name: str = DEFAULT_MODEL):
    """
    Given a comma-separated tech_stack string, call the Gemini model to generate
    3-5 screening questions per technology. Returns a dict: { tech: [questions...] }.

    Raises ValueError on failure.
    """

    if not tech_stack or not tech_stack.strip():
        raise ValueError("Empty tech stack provided.")

    # Normalize tech list
    techs = [t.strip() for t in tech_stack.split(",") if t.strip()]
    if not techs:
        raise ValueError("No valid technologies parsed from tech_stack.")

    # Prompt that requests strict JSON output
    prompt = f"""
You are TalentScout's technical interviewer assistant. Output ONLY valid JSON (no extra text).
Given the tech list below, produce between {min_q} and {max_q} technical screening questions for EACH technology.
For each technology produce an array of question objects with fields:
- question (string)
- difficulty (one of: "junior", "mid", "senior")
- expected_keywords (array of short keywords)
- sample_answer (1-2 short sentences)

Return JSON exactly in this format:
{{
  "questions": {{
    "TechName1": [
      {{"question":"...","difficulty":"mid","expected_keywords":["..."],"sample_answer":"..."}},
      ...
    ],
    "TechName2": [...],
    ...
  }},
  "summary": "one-line summary for interviewer"
}}

Tech list:
{json.dumps(techs)}
"""

    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = _extract_text_from_response(resp)

        # Try parse JSON
        data = _parse_json_from_text(text)
        if data and isinstance(data, dict) and "questions" in data:
            # ensure mapping only contains requested tech names
            parsed_questions = {}
            for t in techs:
                if t in data.get("questions", {}):
                    parsed_questions[t] = [q.get("question", "").strip() for q in data["questions"][t] if q.get("question")]
                else:
                    # ignore if missing; fallback will try to distribute
                    parsed_questions[t] = []
            # If some techs are empty, try filling via fallback parsing of the raw text
            if any(len(v) == 0 for v in parsed_questions.values()):
                fallback = _parse_text_by_headings(techs, text)
                for t in techs:
                    if not parsed_questions.get(t):
                        parsed_questions[t] = fallback.get(t, [])
            return parsed_questions

        # If JSON parsing failed, try heading-based parser
        fallback = _parse_text_by_headings(techs, text)
        if any(len(v) > 0 for v in fallback.values()):
            return fallback

        # As a last resort, look for plain lines and distribute
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Remove disclaimers
        lines = [re.sub(r'^\d+\.\s*', '', ln) for ln in lines if len(ln) > 5]
        if lines:
            per = max(min_q, min(max_q, (len(lines) + len(techs) - 1) // len(techs)))
            distributed = {t: [] for t in techs}
            idx = 0
            for ln in lines:
                distributed[techs[idx % len(techs)]].append(ln)
                idx += 1
            for t in techs:
                distributed[t] = distributed[t][:max_q]
            return distributed

        raise ValueError("Could not parse useful questions from model response.")

    except Exception as exc:
        # Re-raise as ValueError for caller to present user-friendly message
        raise ValueError(f"LLM generation failed: {exc}")
