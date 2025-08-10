"""
app.py - TalentScout Hiring Assistant Main Application

This module implements the main Streamlit interface for the TalentScout Hiring Assistant.
The chatbot guides candidates through three phases:
1. Info Collection: Gather personal and professional details
2. Technical Questions: Ask tailored questions based on tech stack
3. Completion: Summary and next steps

Features:
- Exit keyword detection throughout the conversation
- Comprehensive fallback mechanisms
- Sentiment analysis of candidate responses
- Secure data handling with PII anonymization
"""

import streamlit as st
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional

# Import our custom modules
from prompts import generate_tech_questions, analyze_sentiment
from utils import save_submission, is_exit_keyword, validate_email, validate_phone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EXIT_KEYWORDS = {"exit", "quit", "bye", "stop", "end", "thanks", "thank you", "done"}
OFF_TOPIC_TRIGGERS = [
    "what time", "what's the time", "time is it", "what is the time",
    "how are you", "who are you", "tell me a joke", "weather", "politics", 
    "who is", "current events", "news", "stock market", "sports"
]

def initialize_session_state():
    """Initialize all session state variables with default values."""
    session_defaults = {
        "candidate_data": {},
        "questions_by_tech": {},
        "question_queue": [],
        "current_q_index": 0,
        "responses": [],
        "conversation_phase": "info",
        "last_shown_tech": None,
        "questions_generated": False,
        "sentiment_scores": [],
        "error_count": 0,
        "warning_shown": False
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def check_exit_intent(user_input: str) -> bool:
    """
    Check if user input contains exit keywords or intent to leave.
    
    Args:
        user_input (str): The user's input text
        
    Returns:
        bool: True if exit intent detected, False otherwise
    """
    if not user_input:
        return False
    
    return is_exit_keyword(user_input.strip())

def handle_exit_conversation():
    """Handle graceful exit from conversation at any point."""
    st.session_state.conversation_phase = "done"
    st.info("👋 Thank you for your time! Exiting the conversation...")
    st.rerun()

def show_progress_info():
    """Display current progress information to the user."""
    queue = st.session_state.question_queue
    idx = st.session_state.current_q_index
    
    if queue:
        progress = (idx + 1) / len(queue)
        st.progress(progress)
        st.caption(f"Progress: {idx + 1} / {len(queue)} questions completed")

def handle_api_error(error: Exception, context: str = "API call"):
    """
    Handle API errors with user-friendly messages and fallback options.
    
    Args:
        error (Exception): The exception that occurred
        context (str): Context where the error occurred
    """
    st.session_state.error_count += 1
    logger.error(f"API Error in {context}: {str(error)}")
    
    if st.session_state.error_count >= 3:
        st.error("⚠️ We're experiencing technical difficulties. Please try again later or contact support.")
        if st.button("Reset and Try Again"):
            st.session_state.error_count = 0
            st.session_state.conversation_phase = "info"
            st.rerun()
    else:
        st.error(f"❌ {context} failed. Please try again. (Attempt {st.session_state.error_count}/3)")

def display_sentiment_feedback(sentiment_score: float):
    """
    Display sentiment feedback to help improve candidate experience.
    
    Args:
        sentiment_score (float): Sentiment score between -1 (negative) and 1 (positive)
    """
    if sentiment_score < -0.5:
        st.info("💭 I notice you might be feeling uncertain. Take your time with the questions!")
    elif sentiment_score > 0.7:
        st.success("😊 Great enthusiasm! Keep up the positive energy!")

# --- Page Configuration ---
st.set_page_config(
    page_title="TalentScout Hiring Assistant", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
initialize_session_state()

# --- Header Section ---
st.title("🤖 TalentScout Hiring Assistant")
st.markdown("""
**Welcome to TalentScout's AI-powered hiring assistant!** 

I'll help streamline your application process by:
- 📋 Collecting your professional details
- 🔧 Generating technical questions based on your skills
- 💡 Providing an interactive screening experience

**Privacy Note:** Your data is handled securely and stored anonymized locally.

---
💡 **Tip:** Type 'exit', 'quit', or 'bye' at any time to end the conversation.
""")

# --- Phase 1: Information Collection ---
if st.session_state.conversation_phase == "info":
    st.header("📋 Let's Get Started")
    st.write("Please provide your details below to generate personalized technical questions.")
    
    with st.form("candidate_form", clear_on_submit=False):
        # Input fields with better validation
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input(
                "Full Name *", 
                placeholder="Enter your full name",
                help="This will be anonymized in our records"
            )
            email = st.text_input(
                "Email Address *", 
                placeholder="your.email@domain.com",
                help="We'll use this to contact you"
            )
            phone = st.text_input(
                "Phone Number", 
                placeholder="+1234567890",
                help="Optional - for urgent communications"
            )
        
        with col2:
            experience = st.number_input(
                "Years of Experience *", 
                min_value=0, 
                max_value=50, 
                step=1,
                help="Total years of professional experience"
            )
            desired_position = st.text_input(
                "Desired Position(s)", 
                placeholder="e.g., Software Engineer, Data Scientist",
                help="What role(s) are you interested in?"
            )
            location = st.text_input(
                "Current Location", 
                placeholder="City, Country",
                help="Your current location"
            )
        
        tech_stack = st.text_area(
            "Tech Stack *", 
            placeholder="Python, React, PostgreSQL, Docker, AWS...",
            help="List technologies, languages, frameworks you know (comma-separated)",
            height=100
        )
        
        # Check for exit keywords in form inputs
        form_inputs = [full_name, email, phone, desired_position, location, tech_stack]
        if any(check_exit_intent(inp) for inp in form_inputs if inp):
            st.warning("👋 Detected exit intent. Click below to confirm.")
            if st.form_submit_button("Exit Conversation", type="secondary"):
                handle_exit_conversation()
        
        submitted = st.form_submit_button("🚀 Submit and Generate Questions", type="primary")
    
    if submitted:
        # Comprehensive validation
        errors = []
        
        if not full_name.strip():
            errors.append("Full Name is required")
        elif check_exit_intent(full_name):
            handle_exit_conversation()
            
        if not email.strip():
            errors.append("Email Address is required")
        elif not validate_email(email.strip()):
            errors.append("Please enter a valid email address")
        elif check_exit_intent(email):
            handle_exit_conversation()
            
        if phone.strip() and not validate_phone(phone.strip()):
            errors.append("Please enter a valid phone number")
            
        if not tech_stack.strip():
            errors.append("Tech Stack is required")
        elif check_exit_intent(tech_stack):
            handle_exit_conversation()
            
        if len(tech_stack.strip()) < 10:
            errors.append("Please provide more details about your tech stack")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            # Store candidate data
            st.session_state.candidate_data = {
                "full_name": full_name.strip(),
                "email": email.strip(),
                "phone": phone.strip(),
                "experience": int(experience),
                "desired_position": desired_position.strip(),
                "location": location.strip(),
                "tech_stack": tech_stack.strip(),
                "started_at": datetime.utcnow().isoformat() + "Z"
            }
            
            # Save initial data
            try:
                save_submission(st.session_state.candidate_data)
                logger.info(f"Candidate data saved for {email}")
            except Exception as e:
                logger.error(f"Failed to save candidate data: {e}")
                st.warning("⚠️ Data saving issue, but continuing with questions...")
            
            # Generate questions
            with st.spinner("🔄 Generating personalized technical questions..."):
                try:
                    questions_by_tech = generate_tech_questions(tech_stack.strip())
                    
                    # Create question queue
                    queue = []
                    for tech in questions_by_tech:
                        for q in questions_by_tech[tech]:
                            queue.append({"tech": tech, "question": q})
                    
                    if not queue:
                        st.error("❌ Could not generate questions. Please check your tech stack and try again.")
                        st.info("💡 Try simplifying your tech stack (e.g., 'Python, JavaScript, SQL')")
                    else:
                        st.session_state.questions_by_tech = questions_by_tech
                        st.session_state.question_queue = queue
                        st.session_state.current_q_index = 0
                        st.session_state.conversation_phase = "questions"
                        st.session_state.questions_generated = True
                        
                        st.success(f"✅ Generated {len(queue)} personalized questions!")
                        st.info("🎯 Starting your technical screening...")
                        st.rerun()
                        
                except Exception as e:
                    handle_api_error(e, "Question generation")

# --- Phase 2: Technical Questions ---
elif st.session_state.conversation_phase == "questions":
    queue = st.session_state.question_queue
    idx = st.session_state.current_q_index
    
    # Check if all questions completed
    if idx >= len(queue):
        st.session_state.conversation_phase = "done"
        st.rerun()
    
    current = queue[idx]
    tech = current["tech"]
    question_text = current["question"]
    
    # Display tech section header
    if st.session_state.last_shown_tech != tech:
        st.header(f"🔧 {tech} Questions")
        st.markdown(f"Let's explore your **{tech}** expertise!")
        st.session_state.last_shown_tech = tech
    
    # Display current question
    st.subheader(f"Question {idx + 1} of {len(queue)}")
    st.markdown(f"**{question_text}**")
    
    # Answer input with exit detection
    answer_key = f"answer_{idx}"
    
    answer = st.text_area(
        "Your Answer:",
        key=f"textarea_{answer_key}",
        height=120,
        placeholder="Type your answer here... (or 'exit' to finish early)",
        help="Provide a detailed answer to showcase your knowledge"
    )
    
    # Check for exit keywords in answer
    if answer and check_exit_intent(answer):
        st.warning("👋 Detected exit request. Do you want to end the screening?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Exit", key=f"confirm_exit_{idx}"):
                handle_exit_conversation()
        with col2:
            if st.button("No, Continue", key=f"continue_{idx}"):
                st.rerun()
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_btn = st.button("✅ Submit Answer", key=f"submit_{idx}", type="primary")
    with col2:
        skip_btn = st.button("⏭️ Skip", key=f"skip_{idx}")
    with col3:
        exit_btn = st.button("🚪 Exit", key=f"exit_{idx}", type="secondary")
    
    if exit_btn:
        handle_exit_conversation()
    
    if submit_btn:
        ans = answer.strip()
        
        if not ans:
            st.warning("⚠️ Please provide an answer or click 'Skip' to move on.")
        elif len(ans) < 10:
            st.warning("⚠️ Your answer seems quite short. Could you provide more detail?")
        elif any(trigger in ans.lower() for trigger in OFF_TOPIC_TRIGGERS):
            st.info("🎯 Let's stay focused on the technical question. Please provide a relevant answer.")
        else:
            # Analyze sentiment
            try:
                sentiment_score = analyze_sentiment(ans)
                st.session_state.sentiment_scores.append(sentiment_score)
                display_sentiment_feedback(sentiment_score)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                sentiment_score = 0.0
            
            # Record answer
            st.session_state.responses.append({
                "tech": tech,
                "question": question_text,
                "answer": ans,
                "sentiment_score": sentiment_score,
                "answered_at": datetime.utcnow().isoformat() + "Z"
            })
            
            st.success("✅ Answer recorded! Moving to the next question...")
            
            # Move to next question
            st.session_state.current_q_index += 1
            next_idx = st.session_state.current_q_index
            if next_idx < len(queue) and queue[next_idx]["tech"] != tech:
                st.session_state.last_shown_tech = None
            
            st.rerun()
    
    if skip_btn:
        # Record skipped question
        st.session_state.responses.append({
            "tech": tech,
            "question": question_text,
            "answer": "[SKIPPED]",
            "sentiment_score": 0.0,
            "answered_at": datetime.utcnow().isoformat() + "Z"
        })
        
        st.info("⏭️ Question skipped. Moving to the next one...")
        st.session_state.current_q_index += 1
        next_idx = st.session_state.current_q_index
        if next_idx < len(queue) and queue[next_idx]["tech"] != tech:
            st.session_state.last_shown_tech = None
        
        st.rerun()
    
    # Show progress
    show_progress_info()
    
    # Sidebar with helpful info
    with st.sidebar:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific and detailed
        - Include examples when possible
        - Explain your thought process
        - Don't worry if you're unsure
        """)
        
        st.markdown("### 📊 Your Progress")
        completed = len([r for r in st.session_state.responses if r.get('answer') != '[SKIPPED]'])
        skipped = len([r for r in st.session_state.responses if r.get('answer') == '[SKIPPED]'])
        
        st.metric("Completed", completed)
        st.metric("Skipped", skipped)
        
        if st.session_state.sentiment_scores:
            avg_sentiment = sum(st.session_state.sentiment_scores) / len(st.session_state.sentiment_scores)
            sentiment_label = "😊 Positive" if avg_sentiment > 0.2 else "😐 Neutral" if avg_sentiment > -0.2 else "😟 Negative"
            st.metric("Overall Mood", sentiment_label)

# --- Phase 3: Completion ---
elif st.session_state.conversation_phase == "done":
    # Save final results
    try:
        final_record = st.session_state.candidate_data.copy()
        final_record["responses"] = st.session_state.responses
        final_record["sentiment_scores"] = st.session_state.sentiment_scores
        final_record["completed_at"] = datetime.utcnow().isoformat() + "Z"
        save_submission(final_record)
        logger.info("Final candidate record saved successfully")
    except Exception as e:
        logger.error(f"Failed to save final record: {e}")
        st.warning("⚠️ There was an issue saving your responses, but don't worry - we have your information.")
    
    # Success message
    st.balloons()
    st.success("🎉 Screening Complete! Thank you for your time.")
    
    st.markdown("""
    ### 🚀 What Happens Next?
    
    1. **Review Process**: Our technical team will review your responses
    2. **Evaluation**: We'll assess your technical skills and cultural fit
    3. **Follow-up**: You'll hear from us within 3-5 business days
    4. **Next Steps**: If selected, we'll schedule a detailed technical interview
    
    ### 📧 Stay Connected
    - Check your email for updates
    - Follow us on LinkedIn for job opportunities
    - Visit our careers page for other open positions
    """)
    
    # Session Summary
    with st.expander("📊 Your Session Summary", expanded=False):
        total_questions = len(st.session_state.question_queue)
        answered = len([r for r in st.session_state.responses if r.get('answer') != '[SKIPPED]'])
        skipped = total_questions - answered
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Technologies Covered", len(st.session_state.questions_by_tech))
        with col2:
            st.metric("Questions Answered", answered)
        with col3:
            st.metric("Questions Skipped", skipped)
        
        if st.session_state.sentiment_scores:
            avg_sentiment = sum(st.session_state.sentiment_scores) / len(st.session_state.sentiment_scores)
            st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
        
        st.markdown("**Technologies Evaluated:**")
        for tech in st.session_state.questions_by_tech.keys():
            st.write(f"• {tech}")
    
    # Reset option
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Start New Session", type="secondary"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        st.markdown("[🌐 Visit TalentScout Careers](https://talentscout.com/careers)")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 TalentScout Hiring Assistant v1.0 | 
    Built with ❤️ using Streamlit | 
    🔒 Your data is secure and anonymized
    </div>
    """, 
    unsafe_allow_html=True
)