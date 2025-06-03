# TEXT-SUMMARIZATION-TOOL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SOURAV PAL CHAUDHURI

*ITERN ID*: CT08DL242

*DOMAIN*: AI

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH


# 📄 Article Summarization Tool

A comprehensive Python-based text summarization system that implements multiple algorithms to automatically generate concise summaries from long articles and documents. No external NLP libraries like NLTK required - uses scikit-learn and NetworkX for advanced text processing.

## ✨ Features

- **Multiple Summarization Algorithms**: Frequency-based, TF-IDF, and TextRank methods
- **Smart Sentence Scoring**: Position weighting, length optimization, and content relevance
- **Advanced Text Processing**: Custom tokenization, stop word filtering, and text cleaning
- **Interactive Mode**: Command-line interface for custom text summarization
- **Comprehensive Analysis**: Text statistics and compression ratio calculations
- **Robust Error Handling**: Graceful fallbacks between different methods
- **No NLTK Dependency**: Self-contained implementation with minimal external dependencies

## 🛠️ Installation

### Required Dependencies
```bash
pip install numpy networkx scikit-learn
```

### System Requirements
- Python 3.6+
- NumPy for numerical computations
- NetworkX for graph-based algorithms
- scikit-learn for TF-IDF vectorization

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install numpy networkx scikit-learn
   ```

2. **Run the tool**:
   ```bash
   python article_summarizer.py
   ```

3. **Use in your code**:
   ```python
   from article_summarizer import ArticleSummarizer
   
   summarizer = ArticleSummarizer()
   summary = summarizer.summarize(your_text, method='tfidf', num_sentences=3)
   ```

## 📖 Usage

### Command Line Interface

The tool provides two modes:

1. **Demonstration Mode** - Shows sample article summaries using different algorithms
2. **Interactive Mode** - Allows you to input custom text for summarization

### Interactive Menu Options

- Enter custom text for summarization
- Choose from 3 different algorithms
- Specify number of sentences in summary (1-10)
- View detailed statistics and compression ratios

### Supported Methods

```python
# Available summarization methods
methods = ['frequency', 'tfidf', 'textrank']

# Generate summary
summary = summarizer.summarize(text, method='tfidf', num_sentences=3)
```

## 🎯 Summarization Algorithms

| Algorithm | Description | Best For | Accuracy |
|-----------|-------------|----------|----------|
| **Frequency-based** | Word frequency with position weighting | General purpose, fast | Medium |
| **TF-IDF** | Term frequency-inverse document frequency | Technical documents | High |
| **TextRank** | Graph-based PageRank algorithm | Complex articles | High |

### Algorithm Details

#### 1. Frequency-Based Summarization
- Calculates normalized word frequencies
- Applies position-based scoring (first/last sentences prioritized)
- Considers sentence length optimization
- Fast and reliable for general content

#### 2. TF-IDF Summarization
- Uses scikit-learn's TfidfVectorizer
- Includes bigrams for better context understanding
- Filters stop words automatically
- Excellent for technical and academic content

#### 3. TextRank Summarization
- Implements PageRank algorithm on sentence similarity graph
- Uses Jaccard and cosine similarity metrics
- Builds network graph with NetworkX
- Best for complex, interconnected content

## 🔧 Key Features

### Advanced Text Processing
- **Smart Tokenization**: Handles abbreviations (Dr., Mr., etc.)
- **URL/Email Removal**: Cleans web-scraped content
- **Stop Word Filtering**: Comprehensive 200+ stop words list
- **Sentence Boundary Detection**: Advanced regex patterns

### Intelligent Scoring System
- **Position Weighting**: First and last sentences get higher scores
- **Length Optimization**: Prefers medium-length sentences
- **Content Relevance**: Multiple similarity metrics
- **Normalization**: Balanced scoring across different text types

### Text Statistics
```python
# Automatic analysis includes:
- Total characters and words
- Sentence count
- Average words per sentence
- Average characters per word
- Compression ratio
- Reduction percentage
```

## 🔍 Error Handling

### Robust Fallback System
- TF-IDF failure → Falls back to frequency method
- TextRank failure → Falls back to frequency method
- Short text handling → Returns original if too brief
- Invalid method → Clear error message with valid options

### Input Validation
- Minimum text length requirements
- Sentence count limits (1-10)
- Method name validation
- Graceful handling of edge cases

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install numpy networkx scikit-learn
```

**2. Memory Issues with Large Texts**
```python
# Process in chunks for very large documents
# Recommended: <50,000 words per summarization
```

**3. Poor Summary Quality**
```python
# Try different methods for different content types:
# - Technical content: use 'tfidf'
# - General articles: use 'frequency' 
# - Complex narratives: use 'textrank'
```

## 📋 Requirements

### Core Dependencies
- `numpy >= 1.19.0` - Numerical computations
- `networkx >= 2.5` - Graph algorithms for TextRank
- `scikit-learn >= 0.24.0` - TF-IDF vectorization

### Optional Enhancements
- `matplotlib` - For visualization of text statistics
- `pandas` - For batch processing of multiple documents
- `nltk` - If you want to extend with additional NLP features


## OUTPUT

![image](https://github.com/user-attachments/assets/a56cfe0d-58d8-45fe-bdac-7746ad3ee840)
