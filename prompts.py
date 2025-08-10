"""
prompts.py - AI/ML Question Generation and Sentiment Analysis Module

This module handles:
1. Technical question generation using Google Gemini API based on candidate's tech stack
2. Sentiment analysis of candidate responses using TextBlob
3. Prompt engineering for consistent, high-quality question generation
4. Fallback mechanisms for robust question parsing

Features:
- Intelligent tech stack parsing and question distribution
- Multiple fallback parsing strategies for robust question extraction
- Sentiment analysis for candidate experience optimization
- Secure API key handling with environment variables
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration - Use environment variable for security
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    logger.error("GEMINI_API_KEY not found in environment. Please set it before running.")
    raise ValueError("Missing Gemini API key in environment variables.")

try:
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

# Constants
DEFAULT_MODEL = "gemini-1.5-flash"
MAX_TOTAL_QUESTIONS = 5
MIN_QUESTION_LENGTH = 10
MAX_RETRIES = 3

# Question difficulty levels and their characteristics
DIFFICULTY_LEVELS = {
    "junior": "Basic concepts and syntax",
    "mid": "Practical application and problem-solving",
    "senior": "Advanced concepts, architecture, and optimization"
}

def analyze_sentiment(text: str) -> float:
    """
    Analyze the sentiment of candidate responses using TextBlob.
    
    This helps us understand candidate engagement and emotional state
    during the screening process.
    
    Args:
        text (str): The candidate's response text
        
    Returns:
        float: Sentiment polarity score between -1 (negative) and 1 (positive)
        
    Example:
        >>> analyze_sentiment("I love working with Python!")
        0.625
        >>> analyze_sentiment("This is confusing and difficult")
        -0.4
    """
    try:
        if not text or len(text.strip()) < 5:
            return 0.0
            
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        logger.debug(f"Sentiment analysis: '{text[:50]}...' -> {sentiment_score}")
        return round(sentiment_score, 3)
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return 0.0

def _extract_text_from_response(resp) -> str:
    """
    Extract plain text from Gemini response object.
    
    Handles different SDK versions and response formats robustly.
    
    Args:
        resp: Gemini API response object
        
    Returns:
        str: Extracted text content
    """
    if resp is None:
        return ""
    
    # Try common response attributes
    if hasattr(resp, 'text') and resp.text:
        return resp.text
    
    if hasattr(resp, 'candidates') and resp.candidates:
        candidate = resp.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                return candidate.content.parts[0].text
    
    # Fallback to string conversion
    try:
        return str(resp)
    except Exception as e:
        logger.warning(f"Failed to extract text from response: {e}")
        return ""

def _parse_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract and parse JSON from potentially noisy text response.
    
    Uses multiple strategies to find and parse JSON content.
    
    Args:
        text (str): Raw text that may contain JSON
        
    Returns:
        Optional[Dict]: Parsed JSON object or None if parsing fails
    """
    if not text:
        return None
    
    # Strategy 1: Find JSON block with braces
    json_match = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed for matched block: {e}")
    
    # Strategy 2: Try parsing entire text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")
    
    # Strategy 3: Clean and retry
    cleaned_text = re.sub(r'^[^{]*', '', text)  # Remove prefix
    cleaned_text = re.sub(r'[^}]*$', '', cleaned_text)  # Remove suffix
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    logger.warning("All JSON parsing strategies failed")
    return None

def _parse_questions_by_headings(techs: List[str], text: str) -> Dict[str, List[str]]:
    """
    Fallback parser: Extract questions using technology headings.
    
    Looks for patterns like "Python:", "JavaScript -", etc. and collects
    subsequent lines as questions until the next heading.
    
    Args:
        techs (List[str]): List of technologies to look for
        text (str): Raw text to parse
        
    Returns:
        Dict[str, List[str]]: Technology -> Questions mapping
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {tech: [] for tech in techs}
    
    current_tech = None
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line is a technology heading
        matched_tech = None
        for tech in techs:
            # Various heading patterns
            patterns = [
                rf'^{re.escape(tech)}\s*:',  # "Python:"
                rf'^{re.escape(tech)}\s*-',  # "Python -"
                rf'^{re.escape(tech)}$',     # "Python"
                rf'^\d+\.\s*{re.escape(tech)}',  # "1. Python"
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns):
                matched_tech = tech
                break
        
        if matched_tech:
            current_tech = matched_tech
            i += 1
            continue
        
        # If we have a current tech, try to add this line as a question
        if current_tech:
            # Clean the line (remove numbering, bullets)
            cleaned = re.sub(r'^\d+\.\s*', '', line)
            cleaned = re.sub(r'^[-•]\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Validate question quality
            if (len(cleaned) >= MIN_QUESTION_LENGTH and 
                not cleaned.lower().startswith(('note:', 'hint:', 'example:'))):
                result[current_tech].append(cleaned)
        
        i += 1
    
    # If no questions found using headings, distribute lines among technologies
    if all(len(questions) == 0 for questions in result.values()):
        logger.info("No heading-based questions found, using line distribution")
        return _distribute_lines_among_techs(techs, lines)
    
    # Limit questions per technology
    max_per_tech = max(1, MAX_TOTAL_QUESTIONS // len(techs))
    for tech in result:
        result[tech] = result[tech][:max_per_tech]
    
    return result

def _distribute_lines_among_techs(techs: List[str], lines: List[str]) -> Dict[str, List[str]]:
    """
    Distribute question lines evenly among technologies as final fallback.
    
    Args:
        techs (List[str]): List of technologies
        lines (List[str]): List of potential question lines
        
    Returns:
        Dict[str, List[str]]: Technology -> Questions mapping
    """
    result = {tech: [] for tech in techs}
    
    # Filter and clean lines
    valid_lines = []
    for line in lines:
        cleaned = re.sub(r'^\d+\.\s*', '', line)
        cleaned = re.sub(r'^[-•]\s*', '', cleaned).strip()
        
        if (len(cleaned) >= MIN_QUESTION_LENGTH and 
            not any(skip in cleaned.lower() for skip in ['note:', 'example:', 'hint:'])):
            valid_lines.append(cleaned)
    
    # Distribute lines round-robin
    if valid_lines:
        questions_per_tech = min(2, len(valid_lines) // len(techs))
        for i, tech in enumerate(techs):
            start_idx = i * questions_per_tech
            end_idx = start_idx + questions_per_tech
            result[tech] = valid_lines[start_idx:end_idx]
    
    return result

def _validate_tech_stack(tech_stack: str) -> List[str]:
    """
    Validate and normalize the technology stack input.
    
    Args:
        tech_stack (str): Comma-separated technology list
        
    Returns:
        List[str]: Validated and normalized technology list
        
    Raises:
        ValueError: If tech stack is invalid
    """
    if not tech_stack or not tech_stack.strip():
        raise ValueError("Technology stack cannot be empty")
    
    # Parse and clean technologies
    techs = [tech.strip() for tech in tech_stack.split(",") if tech.strip()]
    
    if not techs:
        raise ValueError("No valid technologies found in tech stack")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_techs = []
    for tech in techs:
        tech_lower = tech.lower()
        if tech_lower not in seen:
            seen.add(tech_lower)
            unique_techs.append(tech)
    
    # Limit number of technologies
    if len(unique_techs) > MAX_TOTAL_QUESTIONS:
        logger.info(f"Limiting technologies from {len(unique_techs)} to {MAX_TOTAL_QUESTIONS}")
        unique_techs = unique_techs[:MAX_TOTAL_QUESTIONS]
    
    return unique_techs

def generate_tech_questions(
    tech_stack: str, 
    model_name: str = DEFAULT_MODEL
) -> Dict[str, List[str]]:
    """
    Generate technical screening questions based on candidate's technology stack.
    
    This function uses Google Gemini to generate relevant, challenging questions
    for each technology in the candidate's stack. It includes comprehensive
    fallback mechanisms and error handling.
    
    Args:
        tech_stack (str): Comma-separated list of technologies
        model_name (str): Gemini model to use for generation
        
    Returns:
        Dict[str, List[str]]: Technology -> Questions mapping
        
    Raises:
        ValueError: If tech stack is invalid or question generation fails
        
    Example:
        >>> questions = generate_tech_questions("Python, React, PostgreSQL")
        >>> print(questions)
        {
            "Python": ["Explain decorators in Python", "What is GIL?"],
            "React": ["What are React hooks?", "Explain virtual DOM"],
            "PostgreSQL": ["What is ACID compliance?"]
        }
    """
    # Validate input
    try:
        techs = _validate_tech_stack(tech_stack)
        logger.info(f"Generating questions for technologies: {techs}")
    except ValueError as e:
        logger.error(f"Tech stack validation failed: {e}")
        raise
    
    # Calculate optimal question distribution
    total_techs = len(techs)
    questions_per_tech = max(1, MAX_TOTAL_QUESTIONS // total_techs)
    
    # Craft the generation prompt
    prompt = _create_question_generation_prompt(techs, questions_per_tech)
    
    # Attempt question generation with retries
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Question generation attempt {attempt + 1}/{MAX_RETRIES}")
            
            # Call Gemini API
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
            
            # Extract and parse response
            response_text = _extract_text_from_response(response)
            if not response_text:
                raise ValueError("Empty response from API")
            
            logger.debug(f"API response: {response_text[:200]}...")
            
            # Try JSON parsing first
            parsed_data = _parse_json_from_text(response_text)
            if parsed_data and "questions" in parsed_data:
                questions_dict = _process_json_questions(parsed_data["questions"], techs)
                if _validate_questions_dict(questions_dict):
                    logger.info("Successfully generated questions via JSON parsing")
                    return questions_dict
            
            # Fallback to heading-based parsing
            logger.info("JSON parsing failed, trying heading-based parsing")
            questions_dict = _parse_questions_by_headings(techs, response_text)
            if _validate_questions_dict(questions_dict):
                logger.info("Successfully generated questions via heading parsing")
                return questions_dict
            
            # If still no success, log and retry
            logger.warning(f"Attempt {attempt + 1} failed to generate valid questions")
            
        except Exception as e:
            logger.error(f"Question generation attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:  # Last attempt
                # Return basic fallback questions
                logger.warning("All attempts failed, generating fallback questions")
                return _generate_fallback_questions(techs)
    
    # This should never be reached, but just in case
    raise ValueError("Failed to generate questions after all attempts")

def _create_question_generation_prompt(techs: List[str], questions_per_tech: int) -> str:
    """
    Create a well-engineered prompt for question generation.
    
    Args:
        techs (List[str]): Technologies to generate questions for
        questions_per_tech (int): Number of questions per technology
        
    Returns:
        str: Formatted prompt for the LLM
    """
    tech_list = ", ".join(techs)
    
    return f"""You are TalentScout's expert technical interviewer. Generate exactly {questions_per_tech} high-quality screening questions for each technology listed below.

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON - no explanations, no markdown, no extra text
2. Generate exactly {questions_per_tech} question(s) per technology
3. Questions should be practical and assess real-world knowledge
4. Mix difficulty levels: junior (30%), mid (50%), senior (20%)
5. Focus on concepts, problem-solving, and best practices

Required JSON format:
{{
  "questions": {{
    "TechName1": [
      {{"question": "Clear, specific question text", "difficulty": "mid", "expected_keywords": ["keyword1", "keyword2"], "sample_answer": "Brief expected answer"}},
      {{"question": "Another question...", "difficulty": "senior", "expected_keywords": ["concept"], "sample_answer": "Expected response"}}
    ],
    "TechName2": [...]
  }},
  "summary": "Generated {questions_per_tech * len(techs)} questions across {len(techs)} technologies"
}}

Technologies to cover:
{json.dumps(techs)}

Generate practical, interview-ready questions that assess both theoretical knowledge and hands-on experience."""

def _process_json_questions(questions_data: Dict, expected_techs: List[str]) -> Dict[str, List[str]]:
    """
    Process and validate questions from JSON response.
    
    Args:
        questions_data (Dict): Questions data from JSON response
        expected_techs (List[str]): Expected technology list
        
    Returns:
        Dict[str, List[str]]: Processed questions dictionary
    """
    result = {}
    total_questions = 0
    
    for tech in expected_techs:
        tech_questions = []
        
        # Look for questions under this tech (case-insensitive)
        for key, value in questions_data.items():
            if key.lower() == tech.lower():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "question" in item:
                            question_text = item["question"].strip()
                            if len(question_text) >= MIN_QUESTION_LENGTH:
                                tech_questions.append(question_text)
                        elif isinstance(item, str) and len(item.strip()) >= MIN_QUESTION_LENGTH:
                            tech_questions.append(item.strip())
                break
        
        # Limit questions and update totals
        max_allowed = min(2, MAX_TOTAL_QUESTIONS - total_questions)
        tech_questions = tech_questions[:max_allowed]
        result[tech] = tech_questions
        total_questions += len(tech_questions)
        
        if total_questions >= MAX_TOTAL_QUESTIONS:
            break
    
    return result

def _validate_questions_dict(questions_dict: Dict[str, List[str]]) -> bool:
    """
    Validate that the questions dictionary meets quality standards.
    
    Args:
        questions_dict (Dict[str, List[str]]): Questions to validate
        
    Returns:
        bool: True if questions are valid, False otherwise
    """
    if not questions_dict:
        return False
    
    total_questions = sum(len(questions) for questions in questions_dict.values())
    if total_questions == 0:
        return False
    
    # Check that at least 50% of technologies have questions
    techs_with_questions = sum(1 for questions in questions_dict.values() if questions)
    if techs_with_questions < len(questions_dict) * 0.5:
        return False
    
    # Validate question quality
    for tech, questions in questions_dict.items():
        for question in questions:
            if not isinstance(question, str) or len(question.strip()) < MIN_QUESTION_LENGTH:
                return False
    
    return True

def _generate_fallback_questions(techs: List[str]) -> Dict[str, List[str]]:
    """
    Generate basic fallback questions when API fails.
    
    Args:
        techs (List[str]): Technologies to generate questions for
        
    Returns:
        Dict[str, List[str]]: Basic questions for each technology
    """
    logger.info("Generating fallback questions")
    
    # Generic question templates
    templates = [
        "What are the key features and advantages of {}?",
        "Describe a challenging project you've worked on using {}.",
        "What are some common best practices when working with {}?",
        "How would you optimize performance in a {}-based application?",
        "What are some common pitfalls to avoid when using {}?"
    ]
    
    result = {}
    questions_per_tech = max(1, MAX_TOTAL_QUESTIONS // len(techs))
    
    for i, tech in enumerate(techs):
        tech_questions = []
        for j in range(questions_per_tech):
            if j < len(templates):
                question = templates[j].format(tech)
                tech_questions.append(question)
        
        if tech_questions:  # Only add if we have questions
            result[tech] = tech_questions
    
    logger.info(f"Generated {sum(len(q) for q in result.values())} fallback questions")
    return result

# Utility functions for additional features

def get_question_difficulty_distribution(questions_dict: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Analyze the difficulty distribution of generated questions.
    
    Args:
        questions_dict (Dict[str, List[str]]): Generated questions
        
    Returns:
        Dict[str, int]: Difficulty level counts
    """
    # This is a placeholder - in a real implementation, you'd analyze
    # question complexity, keywords, etc. to determine difficulty
    total_questions = sum(len(questions) for questions in questions_dict.values())
    
    return {
        "junior": max(1, total_questions // 3),
        "mid": max(1, total_questions // 2),
        "senior": max(1, total_questions // 4)
    }

def suggest_additional_questions(tech: str, current_questions: List[str]) -> List[str]:
    """
    Suggest additional questions for a specific technology.
    
    Args:
        tech (str): Technology name
        current_questions (List[str]): Currently generated questions
        
    Returns:
        List[str]: Additional question suggestions
    """
    # This could be enhanced with a database of questions or additional API calls
    suggestions = [
        f"How do you handle error management in {tech}?",
        f"What testing strategies do you use with {tech}?",
        f"Describe the ecosystem and tooling around {tech}."
    ]
    
    # Filter out similar questions
    filtered_suggestions = []
    for suggestion in suggestions:
        if not any(word in suggestion.lower() for question in current_questions 
                  for word in question.lower().split()[:3]):  # Avoid similar questions
            filtered_suggestions.append(suggestion)
    
    return filtered_suggestions[:2]  # Return max 2 suggestions
    