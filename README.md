# 🤖 TalentScout Hiring Assistant 

An AI-powered hiring assistant chatbot that streamlines the technical screening process for recruitment agencies. Built with Streamlit and Google Gemini AI, it provides an interactive, personalized candidate evaluation experience.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Google Gemini](https://img.shields.io/badge/google%20gemini-AI-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎥 Demo Video

Watch the TalentScout Hiring Assistant in action:

[![Demo Video](https://cdn.loom.com/sessions/thumbnails/3b3254e2947e47cda2908f11bdb5315a-with-play.gif)](https://www.loom.com/share/3b3254e2947e47cda2908f11bdb5315a?sid=45bcd70a-542a-4c7c-b51f-7dd4a53bca62)

[🎬 View Full Demo](https://www.loom.com/share/3b3254e2947e47cda2908f11bdb5315a?sid=45bcd70a-542a-4c7c-b51f-7dd4a53bca62)

## 🌟 Features

### Core Capabilities
- **Interactive Candidate Screening**: Streamlined 3-phase interview process
- **AI-Powered Question Generation**: Personalized technical questions based on candidate's tech stack
- **Sentiment Analysis**: Real-time mood tracking to optimize candidate experience
- **Secure Data Handling**: PII anonymization and GDPR-compliant storage
- **Graceful Exit Handling**: Smart detection of exit keywords throughout the conversation
- **Comprehensive Fallback Systems**: Robust error handling and recovery mechanisms

### Technical Highlights
- **Multi-phase Conversation Flow**: Info Collection → Technical Questions → Completion
- **Dynamic Question Distribution**: Intelligent allocation based on technology count
- **Real-time Progress Tracking**: Visual progress indicators and completion statistics
- **Atomic Data Operations**: Safe file handling with backup and recovery
- **Input Validation**: Email, phone, and tech stack validation with sanitization

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Git (for cloning)

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/talentscout-hiring-assistant.git
   cd talentscout-hiring-assistant
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

**Get your Gemini API key:**
- Visit [Google AI Studio](https://aistudio.google.com/)
- Create a new project or select existing
- Generate API key from the API section

5. **Run the application**
```bash
   streamlit run app.py
```

The application will open in your browser at http://localhost:8501

## ☁️ AWS Deployment

The application is deployed on AWS EC2 (t2.micro instance) for public access and scalable candidate evaluation.

### AWS Deployment Steps

**Launch EC2 Instance**
- Instance type: t2.micro (eligible for free tier)
- AMI: Amazon Linux 2 or Ubuntu 20.04 LTS
- Security Group: Allow HTTP (port 80) and custom port 8501

## 📋 Usage Guide

### For Candidates

1. **Information Collection Phase**
   - Enter personal details (name, email, phone)
   - Specify years of experience and desired positions
   - List your technology stack (e.g., "Python, React, PostgreSQL, Docker")

2. **Technical Screening Phase**
   - Answer AI-generated questions tailored to your tech stack
   - Use the progress bar to track completion
   - Skip questions if needed or type 'exit' to end early

3. **Completion Phase**
   - Review session summary
   - Receive information about next steps
   - Option to start a new session

### For Recruiters
- **Data Storage**: All submissions stored in submissions.json with anonymized PII
- **Analytics**: View completion rates and candidate statistics
- **Backup System**: Automatic backups prevent data loss

### Exit Keywords
Type any of these to exit gracefully:
exit, quit, bye, stop, end, thanks, done, goodbye

## 🔧 Technical Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.8+ | Core application logic |
| **Frontend** | Streamlit | Interactive web interface |
| **AI Engine** | Google Gemini 1.5 Flash | Question generation & analysis |
| **NLP** | TextBlob | Sentiment analysis |
| **Data Storage** | JSON | Local candidate data storage |
| **Validation** | Regex + Custom logic | Input sanitization |

### Project Structure

```
talentscout-hiring-assistant/
├── app.py                 # Main Streamlit application
├── prompts.py            # AI question generation & sentiment analysis
├── utils.py              # Utility functions & data handling
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # Project documentation
├── submissions.json     # Candidate data storage (created automatically)
└── data/                # Additional data files (created automatically)
```

### Core Modules

#### app.py - Main Application
- **Session Management**: Streamlit session state handling
- **Phase Control**: Three-phase conversation workflow
- **UI Components**: Forms, progress bars, and interactive elements
- **Error Handling**: User-friendly error messages and recovery

#### prompts.py - AI Integration
- **Question Generation**: Dynamic technical questions using Gemini AI
- **Prompt Engineering**: Optimized prompts for consistent output
- **Sentiment Analysis**: Real-time candidate mood assessment
- **Fallback Systems**: Multiple parsing strategies for robustness

#### utils.py - Core Utilities
- **Data Validation**: Email, phone, and input sanitization
- **Security**: PII hashing and anonymization
- **File Operations**: Atomic writes with backup and recovery
- **Exit Detection**: Fuzzy matching for exit intent recognition

## 🎯 Prompt Design Strategy

### Question Generation Prompt

The system uses carefully engineered prompts to ensure high-quality, relevant questions:

```python
# Structured JSON output with metadata
{
  "questions": {
    "Technology": [
      {
        "question": "Specific technical question",
        "difficulty": "junior|mid|senior", 
        "expected_keywords": ["concept1", "concept2"],
        "sample_answer": "Expected response outline"
      }
    ]
  }
}
```

### Key Prompt Engineering Principles

1. **Specificity**: Clear instructions for JSON format and question requirements
2. **Difficulty Distribution**: 30% junior, 50% mid-level, 20% senior questions
3. **Practical Focus**: Real-world scenarios over theoretical knowledge
4. **Fallback Parsing**: Multiple extraction strategies for robust handling
5. **Context Awareness**: Questions tailored to candidate's experience level

### Sentiment Analysis Integration
- **Real-time Feedback**: Encourages candidates when sentiment drops
- **Experience Optimization**: Adjusts tone based on emotional state
- **Data Collection**: Tracks overall candidate satisfaction

## 🛠️ Configuration

### Environment Variables

Create a .env file with:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
LOG_LEVEL=INFO
MAX_QUESTIONS_PER_TECH=2
BACKUP_RETENTION_DAYS=7
```

### Application Settings

Modify constants in modules for customization:

```python
# app.py
MAX_TOTAL_QUESTIONS = 5  # Total questions across all technologies
EXIT_KEYWORDS = {"exit", "quit", "bye", "stop", "end"}

# prompts.py
DEFAULT_MODEL = "gemini-1.5-flash"  # Gemini model version
MAX_RETRIES = 3  # API retry attempts

# utils.py
SUBMISSIONS_FILE = "submissions.json"  # Data storage file
```

## 📊 Data Privacy & Security

### Privacy Features
- **PII Anonymization**: Names, emails, and phone numbers hashed using SHA-256
- **Data Minimization**: Only essential information stored
- **Secure Storage**: Local JSON with backup and recovery systems
- **GDPR Compliance**: Right to be forgotten and data portability

### Security Measures
- **Input Sanitization**: XSS and injection prevention
- **Atomic Operations**: Prevents data corruption during writes
- **Backup System**: Automatic backups with configurable retention
- **Error Isolation**: Graceful degradation without exposing system details

## 🧪 Testing & Validation

### Manual Testing Scenarios

1. **Happy Path**: Complete flow with valid inputs
2. **Edge Cases**: Empty inputs, invalid formats, very long responses
3. **Error Conditions**: API failures, network issues, file corruption
4. **Exit Handling**: Various exit keywords and phrases at different stages
5. **Data Persistence**: Verify data storage and retrieval accuracy

### Validation Features
- **Email Validation**: RFC 5322 compliant regex
- **Phone Validation**: International format support
- **Tech Stack Parsing**: Intelligent comma-separated parsing
- **Response Quality**: Minimum length and content validation

## 🚧 Challenges & Solutions

### Challenge 1: Inconsistent AI Response Format

**Problem**: Gemini API sometimes returns malformed JSON or unexpected formats.

**Solution**:
- Implemented multiple parsing strategies (JSON, heading-based, line distribution)
- Added comprehensive fallback question generation
- Robust error handling with retry mechanisms

```python
# Multiple parsing strategies
def _parse_json_from_text(text: str) -> Optional[Dict]:
    # Strategy 1: JSON block extraction
    # Strategy 2: Direct parsing
    # Strategy 3: Clean and retry
```

### Challenge 2: Secure Data Handling

**Problem**: Need to store candidate data while maintaining privacy compliance.

**Solution**:
- SHA-256 hashing for PII (email, phone, names)
- Atomic file operations with backup systems
- Data anonymization while preserving analytical value

```python
def hash_sensitive_data(data: str) -> str:
    normalized = data.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
```

### Challenge 3: Natural Exit Detection

**Problem**: Users express exit intent in various ways beyond simple keywords.

**Solution**:
- Fuzzy matching with phrase detection
- Context-aware exit recognition
- Graceful conversation termination at any phase

```python
exit_phrases = [
    "i want to exit", "let me quit", "i'm done",
    "that's all", "no more questions"
]
```

### Challenge 4: Question Quality Assurance

**Problem**: Ensuring generated questions are relevant and appropriately challenging.

**Solution**:
- Engineered prompts with difficulty distribution requirements
- Multiple validation layers for question quality
- Fallback question templates for critical failures

### Challenge 5: User Experience Optimization

**Problem**: Maintaining candidate engagement throughout the screening process.

**Solution**:
- Real-time sentiment analysis with encouraging feedback
- Visual progress indicators and completion statistics
- Flexible question skipping and early exit options

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings to all functions
- Include unit tests for new functionality
- Update documentation for API changes
- Ensure backward compatibility where possible

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: International candidate support
- **Video Interview Integration**: AI-powered video screening
- **Advanced Analytics**: Detailed candidate insights and reporting
- **Integration APIs**: Connect with popular ATS systems
- **Machine Learning**: Improved question relevance through learning
- **Real-time Collaboration**: Multi-recruiter evaluation support

### Technical Roadmap
- **Database Migration**: Move from JSON to PostgreSQL/MongoDB
- **Microservices Architecture**: Scalable cloud deployment
- **Authentication System**: Role-based access control
- **API Documentation**: OpenAPI/Swagger integration
- **Performance Optimization**: Caching and query optimization

## 🐛 Troubleshooting

### Common Issues

#### API Key Issues
```bash
Error: GEMINI_API_KEY not found
```
**Solution**: Ensure your .env file contains a valid Gemini API key.

#### Import Errors
```bash
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

#### Permission Errors
```bash
PermissionError: [Errno 13] Permission denied: 'submissions.json'
```
**Solution**: Check file permissions and ensure the application can write to the directory.

#### Memory Issues
**Symptoms**: Slow response, high memory usage
**Solution**:
- Restart the application
- Clear browser cache
- Check for large submission files

## 🙏 Acknowledgments

- **Google AI Team** for the Gemini API
- **Streamlit Community** for the excellent framework

## 📞 Contact

**Project Maintainer**: Abdul Hadi
- Email: abdulhadizeeshan79@gmail.com
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/abdul-hadi-070727259/)
