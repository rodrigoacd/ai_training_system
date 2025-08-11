# AI Training System

A sophisticated teacher-student AI learning platform that uses a large language model (Teacher) to evaluate and guide a smaller model (Student) through iterative training cycles.

## Overview

This system implements a novel approach to AI training where:
- **Teacher AI**: Uses Groq's Llama-3-70B model for evaluation and guidance
- **Student AI**: Local small model (DialoGPT, TinyLlama) that learns from feedback
- **Orchestrator**: Manages training flow and sessions
- **Evaluator**: Provides performance analysis and improvement tracking

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Teacher AI    │    │  Orchestrator   │    │   Student AI    │
│  (Groq Llama)   │◄──►│   (Training     │◄──►│  (Local Model)  │
│                 │    │    Manager)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       │                       ▲
         │                       ▼                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Evaluator     │    │  Data Manager   │    │  JSON Manager   │
│  (Analysis)     │    │  (Q&A Dataset)  │    │  (Logs/Results) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-training-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set up your Groq API key
export GROQ_API_KEY="your-groq-api-key-here"

# Or create a .env file
echo "GROQ_API_KEY=your-groq-api-key-here" > .env
```

### 3. Run the System

```bash
# Start the main application
python main.py

# Or run interactive tests first
python tests/test_interactive.py
```

## Project Structure

```
ai-training-system/
├── config/
│   └── ai_config.py          # AI model configurations
├── src/
│   ├── teacher.py            # Teacher AI (Groq API)
│   ├── student.py            # Student AI (Local model)
│   ├── orchestrator.py       # Training orchestration
│   ├── evaluator.py          # Performance evaluation
│   └── utils/
│       ├── data_manager.py   # Dataset management
│       └── json_manager.py   # Results storage
├── data/
│   ├── sample_qa_dataset.csv # Training questions
│   └── test_questions.csv    # Evaluation questions
├── tests/
│   └── test_interactive.py   # Interactive testing
├── logs/                     # Training logs and results
├── models/                   # Model storage (optional)
├── main.py                   # Main application
└── requirements.txt          # Dependencies
```

## Usage

### Main Menu Options

1. **Run Training Session**: Execute teacher-student training cycles
2. **Run Evaluation Only**: Test student performance without training
3. **Interactive Chat**: Chat directly with the student AI
4. **Interactive Training**: Manual training control
5. **Generate Performance Report**: Create analysis reports with charts
6. **System Configuration**: View and modify settings
7. **Test Dataset**: Validate and explore your dataset

### Training Process

1. **Question Phase**: Student AI receives questions from the dataset
2. **Response Phase**: Student generates answers using its current knowledge
3. **Evaluation Phase**: Teacher AI evaluates responses and provides feedback
4. **Storage Phase**: Results are logged for analysis and improvement tracking
5. **Analysis Phase**: Performance metrics are calculated and stored

### Example Training Session

```python
# Initialize system
orchestrator = TrainingOrchestrator(config)

# Run training with 5 questions
result = orchestrator.run_training_cycle(num_questions=5)

# Check results
if result["success"]:
    print(f"Average Score: {result['statistics']['average_score']:.2f}/10")
    print(f"Corrections Made: {result['statistics']['total_corrections']}")
```

## Configuration

### Teacher AI (Groq)
```python
@dataclass
class TeacherConfig:
    api_key: str = "your-groq-api-key"
    model_name: str = "llama3-70b-8192"
    max_tokens: int = 1024
    temperature: float = 0.3
```

### Student AI (Local)
```python
@dataclass
class StudentConfig:
    model_name: str = "microsoft/DialoGPT-small"
    device: str = "auto"  # auto, cpu, cuda
    max_length: int = 512
    temperature: float = 0.7
```

## Dataset Format

The system expects CSV files with the following format:

```csv
question,answer
"What is the capital of France?","The capital of France is Paris."
"How many continents are there?","There are 7 continents on Earth."
"What is 2 + 2?","2 + 2 equals 4."
```

### Sample Dataset Included

The system comes with a 20-question sample dataset covering:
- General knowledge
- Basic mathematics
- History and literature
- Science and nature
- Geography

## Evaluation & Analysis

### Performance Metrics

- **Score Distribution**: Excellent (9-10), Good (7-8), Fair (5-6), Poor (0-4)
- **Average Scores**: Mean performance across all questions
- **Improvement Tracking**: Before/after training comparisons
- **Response Time Analysis**: Performance speed measurements

### Reports Generated

1. **Text Reports**: Detailed performance analysis
2. **Visual Charts**: Score distributions, improvement trends
3. **Session Logs**: Complete training session records
4. **Comparison Analysis**: Baseline vs post-training results

## Testing

### Interactive Testing

```bash
python tests/test_interactive.py
```

**Test Types Available:**
- Basic Response Test
- Dataset Question Test
- Interactive Question Mode
- Comparison Test
- Model Information Display
- Stress Test (performance)

### Manual Testing

```python
# Test student directly
student = StudentAI(config.student)
response = student.generate_response("What is artificial intelligence?")

# Test teacher evaluation
teacher = TeacherAI(config.teacher)
evaluation = teacher.evaluate_answer(question, correct_answer, student_response)
```

## Monitoring & Logs

### Log Files Structure

```
logs/
├── corrections/
│   └── correction_20231201_143022.json
├── training_logs/
│   └── training_log_20231201_143500.json
└── evaluations/
    └── evaluation_20231201_144000.json
```

### Real-time Monitoring

- Training progress display
- Score tracking per question
- Response time monitoring
- Error logging and handling

## Troubleshooting

### Common Issues

**1. Groq API Connection Failed**
```bash
# Check API key
echo $GROQ_API_KEY

# Test API manually
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://api.groq.com/openai/v1/models
```

**2. Student Model Loading Issues**
- Ensure sufficient RAM/VRAM
- Try smaller models (distilgpt2)
- Check CUDA availability for GPU models

**3. Dataset Loading Problems**
- Verify CSV format and encoding
- Check file permissions
- Ensure proper column names

**4. Poor Student Performance**
- Adjust temperature settings
- Try different base models
- Increase context length
- Review question complexity

## Advanced Features

### Custom Models

```python
# Use different student models
config.student.model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
config.student.model_name = "microsoft/DialoGPT-medium"
config.student.model_name = "distilgpt2"
```

### Batch Processing

```python
# Evaluate multiple Q&A pairs
evaluations = [
    (question1, correct1, student1),
    (question2, correct2, student2),
    # ...
]
results = teacher.batch_evaluate(evaluations)
```

### Custom Evaluation Criteria

```python
# Modify teacher templates
teacher.correction_templates["evaluation"] = """
Custom evaluation prompt here...
Score based on: accuracy, completeness, clarity
"""
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for Teacher AI)
- Optional: CUDA-compatible GPU

### API Requirements
- Groq API key (free tier available)
- Sufficient API credits for training sessions

### Dependencies
See `requirements.txt` for complete list:
- torch, transformers (AI models)
- groq (API client)  
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Groq for providing powerful LLM API
- Hugging Face for transformer models
- The open-source AI community

## 📞 Support

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](your-repo-url/issues)
- 📖 Documentation: [Wiki](your-repo-url/wiki)

---

**Happy Training!** 🎓🤖