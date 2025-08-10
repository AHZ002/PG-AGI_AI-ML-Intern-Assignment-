# #APP.PY

# """
# app.py
# Streamlit UI + flow control.
# Phases:
#  - info: collect candidate details
#  - questions: ask one tech-at-a-time questions (record answers)
#  - done: save results and show next steps
# """

# import streamlit as st
# from prompts import generate_tech_questions
# from utils import save_submission
# from datetime import datetime

# EXIT_KEYWORDS = {"exit", "quit", "bye", "stop", "end", "thanks", "thank you"}

# # --- Page config ---
# st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon="🤖", layout="centered")

# # --- Session state initializations ---
# if "candidate_data" not in st.session_state:
#     st.session_state.candidate_data = {}

# if "questions_by_tech" not in st.session_state:
#     st.session_state.questions_by_tech = {}  # dict: tech -> [q1,q2...]

# if "question_queue" not in st.session_state:
#     st.session_state.question_queue = []  # list of {'tech':..., 'question':...}

# if "current_q_index" not in st.session_state:
#     st.session_state.current_q_index = 0

# if "responses" not in st.session_state:
#     st.session_state.responses = []  # list of {tech, question, answer, answered_at}

# if "conversation_phase" not in st.session_state:
#     st.session_state.conversation_phase = "info"  # info | questions | done

# if "last_shown_tech" not in st.session_state:
#     st.session_state.last_shown_tech = None

# if "questions_generated" not in st.session_state:
#     st.session_state.questions_generated = False

# # --- Greeting / header ---
# st.title("🤖 TalentScout Hiring Assistant")
# st.write("Hello! I'll collect a few details and then ask a short set of technical screening questions based on your tech stack.")
# st.write("Type 'exit' any time to finish early. I will only use your inputs to generate questions and store anonymized results locally.")

# # --- Phase: Info collection ---
# if st.session_state.conversation_phase == "info":
#     with st.form("candidate_form"):
#         full_name = st.text_input("Full Name")
#         email = st.text_input("Email Address")
#         phone = st.text_input("Phone Number")
#         experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
#         desired_position = st.text_input("Desired Position(s)")
#         location = st.text_input("Current Location")
#         tech_stack = st.text_area("Tech Stack (comma separated). Example: Python, Django, PostgreSQL, Docker")

#         submitted = st.form_submit_button("Submit and Generate Questions")

#     if submitted:
#         # Basic validation
#         if not full_name.strip() or not email.strip() or not tech_stack.strip():
#             st.warning("Please enter at least Full Name, Email and Tech Stack to proceed.")
#         else:
#             st.session_state.candidate_data = {
#                 "full_name": full_name.strip(),
#                 "email": email.strip(),
#                 "phone": phone.strip(),
#                 "experience": int(experience),
#                 "desired_position": desired_position.strip(),
#                 "location": location.strip(),
#                 "tech_stack": tech_stack.strip(),
#                 "started_at": datetime.utcnow().isoformat() + "Z"
#             }

#             # Save initial sanitized info (simulated)
#             save_submission(st.session_state.candidate_data)

#             # Generate questions via LLM
#             with st.spinner("Generating tailored technical questions..."):
#                 try:
#                     questions_by_tech = generate_tech_questions(tech_stack)
#                     # flatten into queue (tech order preserved)
#                     queue = []
#                     for tech in questions_by_tech:
#                         for q in questions_by_tech[tech]:
#                             queue.append({"tech": tech, "question": q})
#                     if not queue:
#                         st.error("Could not generate questions. Please try again or simplify the tech stack.")
#                     else:
#                         st.session_state.questions_by_tech = questions_by_tech
#                         st.session_state.question_queue = queue
#                         st.session_state.current_q_index = 0
#                         st.session_state.conversation_phase = "questions"
#                         st.session_state.questions_generated = True
#                         st.success("Questions generated. Starting the interactive screening now.")
#                         st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
#                 except Exception as e:
#                     st.error(f"Failed to generate questions: {e}")

# # --- Phase: Questions (one-by-one) ---
# elif st.session_state.conversation_phase == "questions":
#     queue = st.session_state.question_queue
#     idx = st.session_state.current_q_index

#     if idx >= len(queue):
#         # done
#         st.session_state.conversation_phase = "done"
#         st.rerun()

#     current = queue[idx]
#     tech = current["tech"]
#     question_text = current["question"]

#     # Show tech header once when entering a new technology section
#     if st.session_state.last_shown_tech != tech:
#         st.markdown(f"### Let's start with **{tech}**")
#         st.session_state.last_shown_tech = tech

#     st.markdown(f"**Q{idx + 1}.** {question_text}")

#     # Answer area - Fixed widget key handling
#     answer_key = f"answer_{idx}"
    
#     # Initialize the answer in session state if it doesn't exist
#     if answer_key not in st.session_state:
#         st.session_state[answer_key] = ""
    
#     # Use the text_area widget properly - let Streamlit handle the state
#     answer = st.text_area(
#         "Your answer (type 'exit' to finish):", 
#         value=st.session_state[answer_key], 
#         key=f"textarea_{answer_key}",  # Use a different key for the widget
#         height=160
#     )
    
#     # Update session state with the current value
#     st.session_state[answer_key] = answer

#     col1, col2 = st.columns([1, 1])
#     with col1:
#         submit_btn = st.button("Submit Answer", key=f"submit_{idx}")
#     with col2:
#         skip_btn = st.button("Skip Question", key=f"skip_{idx}")

#     # Off-topic detection triggers (simple heuristic)
#     off_topic_triggers = [
#         "what time", "what's the time", "time is it", "what is the time",
#         "how are you", "who are you", "tell me a joke", "weather", "politics", "who is"
#     ]

#     if submit_btn:
#         ans = answer.strip()
#         if not ans:
#             st.warning("Please write an answer (or click Skip Question).")
#         elif ans.lower() in EXIT_KEYWORDS:
#             st.session_state.conversation_phase = "done"
#             st.rerun()
#         elif any(trigger in ans.lower() for trigger in off_topic_triggers):
#             st.info("I am designed to assist with your application. Let's continue with the question I asked.")
#         elif len(ans) < 8:
#             st.warning("Your answer seems very short — please add a bit more detail to help us assess your skills.")
#         else:
#             # Accept answer, record it
#             st.session_state.responses.append({
#                 "tech": tech,
#                 "question": question_text,
#                 "answer": ans,
#                 "answered_at": datetime.utcnow().isoformat() + "Z"
#             })
#             st.success("Answer recorded. Moving to next question...")
#             st.session_state.current_q_index += 1
#             # clear last_shown_tech if next question is different tech so header shows
#             next_idx = st.session_state.current_q_index
#             if next_idx < len(queue) and queue[next_idx]["tech"] != tech:
#                 st.session_state.last_shown_tech = None
#             st.rerun()

#     if skip_btn:
#         # Record skipped with empty answer marker
#         st.session_state.responses.append({
#             "tech": tech,
#             "question": question_text,
#             "answer": "[SKIPPED]",
#             "answered_at": datetime.utcnow().isoformat() + "Z"
#         })
#         st.warning("Question skipped. Moving to next.")
#         st.session_state.current_q_index += 1
#         next_idx = st.session_state.current_q_index
#         if next_idx < len(queue) and queue[next_idx]["tech"] != tech:
#             st.session_state.last_shown_tech = None
#         st.rerun()

#     # Quick progress info
#     st.progress((idx + 1) / max(1, len(queue)))
#     st.write(f"Progress: {idx + 1} / {len(queue)}")

# # --- Phase: Done / Conclusion ---
# elif st.session_state.conversation_phase == "done":
#     # Final save
#     final_record = st.session_state.candidate_data.copy()
#     final_record["responses"] = st.session_state.responses
#     final_record["completed_at"] = datetime.utcnow().isoformat() + "Z"
#     save_submission(final_record)

#     st.success("That's all the questions I have for you. Thank you for your time!")
#     st.write("Your profile and responses have been recorded. Our recruitment team at TalentScout will review them and get in touch if there's a match.")
#     st.write("Best of luck with your job search!")

#     # Optionally show a short summary (non-PII)
#     st.markdown("#### Session summary (non-PII)")
#     st.write(f"Technologies evaluated: {', '.join(st.session_state.questions_by_tech.keys()) if st.session_state.questions_by_tech else 'N/A'}")
#     st.write(f"Questions asked: {len(st.session_state.question_queue)}")
#     st.write(f"Answers received: {len([r for r in st.session_state.responses if r.get('answer') and r.get('answer') != '[SKIPPED]'])}")

#     # Reset button (for demo/testing)
#     if st.button("Start new session"):
#         for k in list(st.session_state.keys()):
#             del st.session_state[k]
#         st.rerun()

# #PROMPTS.PY

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


# #UTILS.PY

# """
# utils.py
# Helper utilities for hashing PII, loading/saving submissions.
# """

# import json
# import os
# import hashlib
# from datetime import datetime

# SUBMISSIONS_FILE = "submissions.json"


# def hash_sensitive_data(data: str) -> str:
#     """
#     Hash sensitive candidate information (e.g., email, phone)
#     using SHA-256 so raw PII is not stored in plaintext in the sample file.
#     """
#     if not data:
#         return ""
#     return hashlib.sha256(data.strip().lower().encode()).hexdigest()


# def load_submissions():
#     """
#     Load submissions (returns list). If file missing or invalid, returns [].
#     """
#     if not os.path.exists(SUBMISSIONS_FILE):
#         return []
#     try:
#         with open(SUBMISSIONS_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return []


# def save_submission(candidate_data: dict):
#     """
#     Save a candidate submission (anonymizes PII).
#     candidate_data may include fields like:
#       full_name, email, phone, experience, desired_position, location, tech_stack, responses (list)
#     Returns True on success.
#     """
#     submissions = load_submissions()

#     # Create sanitized copy
#     sanitized = candidate_data.copy()

#     # Hash or remove PII
#     if "email" in sanitized:
#         sanitized["email_hash"] = hash_sensitive_data(sanitized.pop("email"))
#     if "phone" in sanitized:
#         sanitized["phone_hash"] = hash_sensitive_data(sanitized.pop("phone"))

#     # Add server-side timestamp
#     sanitized["saved_at"] = datetime.utcnow().isoformat() + "Z"

#     submissions.append(sanitized)

#     # Save back
#     with open(SUBMISSIONS_FILE, "w", encoding="utf-8") as f:
#         json.dump(submissions, f, indent=2, ensure_ascii=False)

#     return True





#cd , env , streamlit run app.py

# fallback error , have to click submit twice , other details

import google.generativeai as genai
genai.configure(api_key="AIzaSyD0Cl-Y9EShSwg9biO0SrvtJ_28vZ8Lzuo")

model = genai.GenerativeModel("gemini-1.5-flash")
resp = model.generate_content("Say hello")
print(resp.text)
