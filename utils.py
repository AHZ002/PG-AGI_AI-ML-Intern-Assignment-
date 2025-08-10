"""
utils.py - Utility Functions for TalentScout Hiring Assistant

This module provides essential utility functions for:
1. Data validation (email, phone, exit keywords)
2. Secure data handling with PII anonymization
3. File operations for candidate data storage
4. Input sanitization and validation

Features:
- Comprehensive input validation
- SHA-256 hashing for sensitive data
- Robust file handling with error recovery
- Exit keyword detection with fuzzy matching
- GDPR-compliant data handling
"""

import json
import os
import hashlib
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUBMISSIONS_FILE = "submissions.json"
BACKUP_FILE = "submissions_backup.json"
EXIT_KEYWORDS = {
    "exit", "quit", "bye", "stop", "end", "thanks", "thank you", "done",
    "goodbye", "good bye", "finish", "complete", "close", "leave",
    "cancel", "abort", "terminate"
}

# Email validation pattern (RFC 5322 compliant)
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?'
    r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
)

# Phone validation pattern (international format)
PHONE_PATTERN = re.compile(r'^\+?[1-9]\d{1,14}$|^[\d\s\-\(\)]{10,}$')

def validate_email(email: str) -> bool:
    """
    Validate email address using RFC 5322 compliant regex.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if valid email, False otherwise
        
    Examples:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
    """
    if not email or not isinstance(email, str):
        return False
    
    email = email.strip().lower()
    
    # Basic length check
    if len(email) < 5 or len(email) > 254:
        return False
    
    # Check for common invalid patterns
    if email.startswith('.') or email.endswith('.') or '..' in email:
        return False
    
    # Use regex validation
    return bool(EMAIL_PATTERN.match(email))

def validate_phone(phone: str) -> bool:
    """
    Validate phone number with support for various international formats.
    
    Args:
        phone (str): Phone number to validate
        
    Returns:
        bool: True if valid phone number, False otherwise
        
    Examples:
        >>> validate_phone("+1234567890")
        True
        >>> validate_phone("(123) 456-7890")
        True
        >>> validate_phone("123")
        False
    """
    if not phone or not isinstance(phone, str):
        return False
    
    # Clean the phone number
    cleaned = re.sub(r'[\s\-\(\)]', '', phone.strip())
    
    # Basic length check (7-15 digits)
    if len(cleaned) < 7 or len(cleaned) > 15:
        return False
    
    # Check if it matches phone patterns
    return bool(PHONE_PATTERN.match(phone.strip()))

def is_exit_keyword(text: str) -> bool:
    """
    Check if text contains exit keywords or phrases with fuzzy matching.
    
    This function detects various ways users might express intent to exit:
    - Direct keywords: "exit", "quit", "bye"
    - Phrases: "i want to exit", "let me quit"
    - Polite forms: "thank you", "that's all"
    
    Args:
        text (str): User input text to check
        
    Returns:
        bool: True if exit intent detected, False otherwise
        
    Examples:
        >>> is_exit_keyword("exit")
        True
        >>> is_exit_keyword("I want to quit now")
        True
        >>> is_exit_keyword("This is my answer")
        False
    """
    if not text or not isinstance(text, str):
        return False
    
    # Normalize text
    normalized = text.lower().strip()
    
    # Direct keyword matching
    if normalized in EXIT_KEYWORDS:
        return True
    
    # Check for exit phrases
    exit_phrases = [
        "i want to exit", "let me exit", "i need to exit",
        "i want to quit", "let me quit", "i need to quit", 
        "i want to leave", "let me leave", "i need to leave",
        "i'm done", "i am done", "that's all", "thats all",
        "no more questions", "stop asking", "end this",
        "i give up", "this is too hard"
    ]
    
    for phrase in exit_phrases:
        if phrase in normalized:
            return True
    
    # Check for single exit words in the text
    words = normalized.split()
    if len(words) <= 3:  # Only check short responses
        for word in words:
            if word in EXIT_KEYWORDS:
                return True
    
    return False

def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive information using SHA-256 for secure storage.
    
    This function ensures PII is not stored in plaintext while maintaining
    the ability to identify unique candidates.
    
    Args:
        data (str): Sensitive data to hash (email, phone, etc.)
        
    Returns:
        str: SHA-256 hash of the input data
        
    Example:
        >>> hash_sensitive_data("user@example.com")
        'b4c9a289323b21a01c3e940f150eb9b8c542587f1abfd8f0e1cc1ffc5e475514'
    """
    if not data or not isinstance(data, str):
        return ""
    
    # Normalize data before hashing
    normalized = data.strip().lower()
    
    # Create hash
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input by removing potential harmful content.
    
    Args:
        text (str): Input text to sanitize
        max_length (int): Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Truncate to max length
    sanitized = text[:max_length]
    
    # Remove potential script injection attempts
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    # Clean up whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return sanitized

def ensure_data_directory():
    """Ensure the data directory exists for storing submissions."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def create_backup(file_path: str) -> bool:
    """
    Create a backup of the submissions file.
    
    Args:
        file_path (str): Path to file to backup
        
    Returns:
        bool: True if backup created successfully
    """
    try:
        if os.path.exists(file_path):
            backup_path = file_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            logger.info(f"Backup created: {backup_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
    return False

def load_submissions() -> List[Dict]:
    """
    Load candidate submissions from JSON file with error recovery.
    
    Returns:
        List[Dict]: List of submission records, empty list if file doesn't exist or is corrupted
        
    Features:
        - Automatic backup creation
        - Corruption recovery
        - Detailed error logging
    """
    try:
        if not os.path.exists(SUBMISSIONS_FILE):
            logger.info(f"Submissions file {SUBMISSIONS_FILE} does not exist, creating new one")
            return []
        
        # Check file size (basic corruption check)
        file_size = os.path.getsize(SUBMISSIONS_FILE)
        if file_size == 0:
            logger.warning("Submissions file is empty")
            return []
        
        # Load the data
        with open(SUBMISSIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate data structure
        if not isinstance(data, list):
            logger.error("Submissions file contains invalid data structure")
            return []
        
        logger.info(f"Successfully loaded {len(data)} submissions")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in submissions file: {e}")
        # Try to recover from backup
        return _recover_from_backup()
    except Exception as e:
        logger.error(f"Unexpected error loading submissions: {e}")
        return []

def _recover_from_backup() -> List[Dict]:
    """
    Attempt to recover data from backup files.
    
    Returns:
        List[Dict]: Recovered submissions or empty list
    """
    try:
        backup_files = [f for f in os.listdir('.') if f.startswith('submissions_backup_') and f.endswith('.json')]
        if backup_files:
            # Use the most recent backup
            latest_backup = max(backup_files, key=lambda x: os.path.getctime(x))
            logger.info(f"Attempting recovery from {latest_backup}")
            
            with open(latest_backup, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                logger.info(f"Successfully recovered {len(data)} submissions from backup")
                return data
    except Exception as e:
        logger.error(f"Backup recovery failed: {e}")
    
    return []

def save_submission(candidate_data: Dict[str, Any]) -> bool:
    """
    Save candidate submission with comprehensive security and error handling.
    
    Features:
    - PII anonymization
    - Data validation
    - Backup creation
    - Atomic writes
    - GDPR compliance
    
    Args:
        candidate_data (Dict[str, Any]): Candidate information and responses
        
    Returns:
        bool: True if saved successfully, False otherwise
        
    Example:
        >>> data = {
        ...     "full_name": "John Doe",
        ...     "email": "john@example.com",
        ...     "tech_stack": "Python, React",
        ...     "responses": [...]
        ... }
        >>> save_submission(data)
        True
    """
    try:
        # Create backup before modifying
        if os.path.exists(SUBMISSIONS_FILE):
            create_backup(SUBMISSIONS_FILE)
        
        # Load existing submissions
        submissions = load_submissions()
        
        # Create sanitized copy of candidate data
        sanitized = _sanitize_candidate_data(candidate_data)
        
        # Add metadata
        sanitized.update({
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "version": "1.0",
            "data_retention_policy": "anonymized_data_kept_for_analysis"
        })
        
        # Add to submissions
        submissions.append(sanitized)
        
        # Atomic write (write to temp file first, then rename)
        temp_file = f"{SUBMISSIONS_FILE}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(submissions, f, indent=2, ensure_ascii=False, default=str)
        
        # Rename temp file to actual file (atomic operation)
        os.replace(temp_file, SUBMISSIONS_FILE)
        
        logger.info(f"Successfully saved submission. Total submissions: {len(submissions)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save submission: {e}")
        
        # Clean up temp file if it exists
        temp_file = f"{SUBMISSIONS_FILE}.tmp"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
                
        return False

def _sanitize_candidate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and anonymize candidate data for secure storage.
    
    Args:
        data (Dict[str, Any]): Raw candidate data
        
    Returns:
        Dict[str, Any]: Sanitized and anonymized data
    """
    sanitized = {}
    
    # Handle each field appropriately
    for key, value in data.items():
        if key == "email" and value:
            sanitized["email_hash"] = hash_sensitive_data(str(value))
            sanitized["email_domain"] = str(value).split('@')[-1] if '@' in str(value) else "unknown"
        elif key == "phone" and value:
            sanitized["phone_hash"] = hash_sensitive_data(str(value))
            sanitized["phone_provided"] = bool(value)
        elif key == "full_name" and value:
            # Store only initials for privacy
            name_parts = str(value).strip().split()
            initials = ''.join([part[0].upper() for part in name_parts if part])
            sanitized["name_initials"] = initials
        elif key in ["responses", "sentiment_scores"]:
            # Keep responses but sanitize if needed
            sanitized[key] = _sanitize_responses(value) if key == "responses" else value
        elif isinstance(value, str):
            sanitized[key] = sanitize_input(str(value))
        else:
            sanitized[key] = value
    
    return sanitized

def _sanitize_responses(responses: List[Dict]) -> List[Dict]:
    """
    Sanitize candidate responses while preserving technical content.
    
    Args:
        responses (List[Dict]): List of response dictionaries
        
    Returns:
        List[Dict]: Sanitized responses
    """
    if not isinstance(responses, list):
        return []
    
    sanitized_responses = []
    for response in responses:
        if isinstance(response, dict):
            sanitized_response = {}
            for key, value in response.items():
                if key == "answer" and isinstance(value, str):
                    # Sanitize answer but keep technical content
                    sanitized_response[key] = sanitize_input(value, max_length=5000)
                else:
                    sanitized_response[key] = value
            sanitized_responses.append(sanitized_response)
    
    return sanitized_responses

def get_submission_stats() -> Dict[str, Any]:
    """
    Get statistics about stored submissions for reporting.
    
    Returns:
        Dict[str, Any]: Statistics about submissions
    """
    try:
        submissions = load_submissions()
        
        if not submissions:
            return {"total": 0, "message": "No submissions found"}
        
        stats = {
            "total_submissions": len(submissions),
            "date_range": {
                "earliest": min(s.get("started_at", "") for s in submissions if s.get("started_at")),
                "latest": max(s.get("started_at", "") for s in submissions if s.get("started_at"))
            },
            "completion_rate": len([s for s in submissions if s.get("completed_at")]) / len(submissions),
            "average_questions_answered": sum(
                len([r for r in s.get("responses", []) if r.get("answer") != "[SKIPPED]"])
                for s in submissions
            ) / len(submissions) if submissions else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate submission stats: {e}")
        return {"error": str(e)}

def cleanup_old_backups(days_to_keep: int = 7) -> int:
    """
    Clean up old backup files to save disk space.
    
    Args:
        days_to_keep (int): Number of days of backups to retain
        
    Returns:
        int: Number of files cleaned up
    """
    try:
        current_time = datetime.now()
        cleanup_count = 0
        
        backup_files = [f for f in os.listdir('.') if f.startswith('submissions_backup_')]
        
        for backup_file in backup_files:
            file_path = Path(backup_file)
            if file_path.exists():
                # Get file age
                file_time = datetime.fromtimestamp(file_path.stat().st_ctime)
                age_days = (current_time - file_time).days
                
                if age_days > days_to_keep:
                    file_path.unlink()
                    cleanup_count += 1
                    logger.info(f"Cleaned up old backup: {backup_file}")
        
        return cleanup_count
        
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")
        return 0