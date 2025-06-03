# TEXT-SUMMARIZATION-TOOL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SOURAV PAL CHAUDHURI

*ITERN ID*: CT08DL242

*DOMAIN*: AI

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH


# ***Article Summarization Tool***

A powerful Python-based text summarization tool that leverages Natural Language Processing (NLP) techniques to generate concise summaries from lengthy articles. This tool implements three different extractive summarization algorithms without relying on NLTK.

**Features**

 -1)Multiple Summarization Methods: Choose from 3 NLP algorithms.  
 
 -2)No NLTK Dependency: Uses only standard Python libraries and scikit-learn.

 -3)Interactive Mode: Real-time summarization with custom input.

 -4)Comprehensive Demo: Pre-loaded sample articles for testing.

 -5)Advanced Text Processing: Intelligent sentence tokenization and preprocessing.

 -6)Statistical Analysis: Detailed compression ratios and text stats.

 -7)Robust Error Handling: Graceful fallbacks between summarization methods.

**Table of Contents**

-1)Installation.  
-2)Quick Start.  
-3)Summarization Methods.  
-4)Usage Examples.  
-5)Performance.

**Installation**

Prerequisites
Python 3.6 or higher

pip package manager

Install Dependencies
bash
pip install scikit-learn numpy networkx
Download the Tool

bash
git clone <repository-url>
cd article-summarization-tool

**Quick Start**

Run the Demo

bash
python article_summarizer.py
This will:

Display summaries of 3 sample articles using all methods

Show compression and performance statistics

Offer interactive mode for custom input text

**Usage Examples**

python
from article_summarizer import ArticleSummarizer

# Initialize the summarizer
summarizer = ArticleSummarizer()

# Your article text
article = "Your long article text here..."

# Generate a summary using the TF-IDF method
summary = summarizer.summarize(article, method='tfidf', num_sentences=3)
print(summary)

**Summarization Methods**
 
1. Frequency-Based Summarization (frequency)
   
How it works:

Calculates word frequency across the text

Scores sentences based on keyword density

Prioritizes first/last sentences and filters by length

Best for: News content, simple articles

python
summary = summarizer.summarize(text, method='frequency', num_sentences=3)

2. TF-IDF Summarization (tfidf)
   
How it works:

Computes Term Frequency-Inverse Document Frequency scores

Selects sentences with high information density

Includes bigram context and position bonuses

Best for: Technical documents, complex texts

python
summary = summarizer.summarize(text, method='tfidf', num_sentences=3)

3. TextRank Summarization (textrank)
   
How it works:

Applies Google's PageRank algorithm to sentence similarity graphs

Uses Jaccard and Cosine similarity for edge weighting

Ranks based on sentence connectivity and position

Best for: Academic, analytical, or interconnected content

python
summary = summarizer.summarize(text, method='textrank', num_sentences=3)


## OUTPUT

![image](https://github.com/user-attachments/assets/a56cfe0d-58d8-45fe-bdac-7746ad3ee840)


**Performance**
Summarization speed and accuracy may vary by method and article type.

Compression ratios and timing are reported for each run.
