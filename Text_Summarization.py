import re
import math
import string
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ArticleSummarizer:
    def __init__(self):
        self.stop_words = self._get_stop_words()
        self.sentence_delimiters = r'[.!?]+(?:\s|$)'
        self.word_pattern = r'\b[a-zA-Z]+\b'
    
    def _get_stop_words(self):
        """Comprehensive stop words list"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'were', 'will', 'with', 'would', 'she', 'her', 'his', 'him', 'they',
            'them', 'their', 'this', 'these', 'those', 'i', 'you', 'we', 'us', 'our',
            'my', 'your', 'me', 'but', 'or', 'if', 'while', 'when', 'where', 'why',
            'how', 'what', 'who', 'which', 'whom', 'whose', 'can', 'could', 'should',
            'would', 'may', 'might', 'must', 'shall', 'will', 'do', 'does', 'did',
            'have', 'had', 'having', 'am', 'being', 'been', 'get', 'got', 'getting',
            'made', 'make', 'making', 'go', 'going', 'went', 'gone', 'come', 'came',
            'coming', 'take', 'took', 'taken', 'taking', 'give', 'gave', 'given',
            'giving', 'put', 'putting', 'see', 'saw', 'seen', 'seeing', 'know',
            'knew', 'known', 'knowing', 'think', 'thought', 'thinking', 'say',
            'said', 'saying', 'tell', 'told', 'telling', 'ask', 'asked', 'asking',
            'work', 'worked', 'working', 'seem', 'seemed', 'seeming', 'feel',
            'felt', 'feeling', 'try', 'tried', 'trying', 'leave', 'left', 'leaving',
            'call', 'called', 'calling', 'keep', 'kept', 'keeping', 'let', 'letting',
            'begin', 'began', 'begun', 'beginning', 'help', 'helped', 'helping',
            'show', 'showed', 'shown', 'showing', 'hear', 'heard', 'hearing',
            'play', 'played', 'playing', 'run', 'ran', 'running', 'move', 'moved',
            'moving', 'live', 'lived', 'living', 'believe', 'believed', 'believing',
            'hold', 'held', 'holding', 'bring', 'brought', 'bringing', 'happen',
            'happened', 'happening', 'write', 'wrote', 'written', 'writing',
            'provide', 'provided', 'providing', 'sit', 'sat', 'sitting', 'stand',
            'stood', 'standing', 'lose', 'lost', 'losing', 'pay', 'paid', 'paying',
            'meet', 'met', 'meeting', 'include', 'included', 'including', 'continue',
            'continued', 'continuing', 'set', 'setting', 'learn', 'learned',
            'learning', 'change', 'changed', 'changing', 'lead', 'led', 'leading',
            'understand', 'understood', 'understanding', 'watch', 'watched',
            'watching', 'follow', 'followed', 'following', 'stop', 'stopped',
            'stopping', 'create', 'created', 'creating', 'speak', 'spoke', 'spoken',
            'speaking', 'read', 'reading', 'allow', 'allowed', 'allowing', 'add',
            'added', 'adding', 'spend', 'spent', 'spending', 'grow', 'grew', 'grown',
            'growing', 'open', 'opened', 'opening', 'walk', 'walked', 'walking',
            'win', 'won', 'winning', 'offer', 'offered', 'offering', 'remember',
            'remembered', 'remembering', 'love', 'loved', 'loving', 'consider',
            'considered', 'considering', 'appear', 'appeared', 'appearing', 'buy',
            'bought', 'buying', 'wait', 'waited', 'waiting', 'serve', 'served',
            'serving', 'die', 'died', 'dying', 'send', 'sent', 'sending', 'expect',
            'expected', 'expecting', 'build', 'built', 'building', 'stay', 'stayed',
            'staying', 'fall', 'fell', 'fallen', 'falling', 'cut', 'cutting',
            'reach', 'reached', 'reaching', 'kill', 'killed', 'killing', 'remain',
            'remained', 'remaining', 'suggest', 'suggested', 'suggesting', 'raise',
            'raised', 'raising', 'pass', 'passed', 'passing', 'sell', 'sold',
            'selling', 'require', 'required', 'requiring', 'report', 'reported',
            'reporting', 'decide', 'decided', 'deciding', 'pull', 'pulled', 'pulling'
        }
    
    def _tokenize_sentences(self, text):
        """Advanced sentence tokenization"""
        # Handle common abbreviations
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'i.e.', 'e.g.']
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split sentences
        sentences = re.split(self.sentence_delimiters, text)
        
        # Restore abbreviations and clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.replace('<DOT>', '.').strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _tokenize_words(self, text):
        """Extract words from text"""
        words = re.findall(self.word_pattern, text.lower())
        return [word for word in words if len(word) > 2 and word not in self.stop_words]
    
    def _preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s.!?,-]', '', text)
        return text.strip()
    
    def _calculate_word_frequencies(self, sentences):
        """Calculate normalized word frequencies"""
        word_freq = Counter()
        
        for sentence in sentences:
            words = self._tokenize_words(sentence)
            for word in words:
                word_freq[word] += 1
        
        # Normalize frequencies
        if word_freq:
            max_freq = max(word_freq.values())
            word_freq = {word: freq / max_freq for word, freq in word_freq.items()}
        
        return word_freq
    
    def _calculate_sentence_position_score(self, sentence_index, total_sentences):
        """Calculate position-based score (first and last sentences are important)"""
        if total_sentences <= 1:
            return 1.0
        
        # Higher scores for beginning and end
        if sentence_index == 0:  # First sentence
            return 1.0
        elif sentence_index == total_sentences - 1:  # Last sentence
            return 0.8
        elif sentence_index == 1:  # Second sentence
            return 0.9
        else:
            # Middle sentences get lower scores
            return 0.5
    
    def _calculate_sentence_length_score(self, sentence, avg_length):
        """Score based on sentence length (prefer medium-length sentences)"""
        length = len(sentence.split())
        if length == 0:
            return 0
        
        # Penalize very short and very long sentences
        if length < 5:
            return 0.3
        elif length > avg_length * 2:
            return 0.6
        else:
            return min(1.0, length / avg_length)
    
    def frequency_based_summary(self, text, num_sentences=3):
        """Generate summary using word frequency scoring with position weighting"""
        text = self._preprocess_text(text)
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        word_freq = self._calculate_word_frequencies(sentences)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = self._tokenize_words(sentence)
            
            # Word frequency score
            word_score = sum(word_freq.get(word, 0) for word in words)
            if len(words) > 0:
                word_score /= len(words)
            
            # Position score
            position_score = self._calculate_sentence_position_score(i, len(sentences))
            
            # Length score
            length_score = self._calculate_sentence_length_score(sentence, avg_length)
            
            # Combine scores
            final_score = (word_score * 0.6) + (position_score * 0.3) + (length_score * 0.1)
            sentence_scores[i] = final_score
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, score in top_sentences[:num_sentences]])
        
        return ' '.join([sentences[i] for i in top_indices])
    
    def tfidf_summary(self, text, num_sentences=3):
        """Generate summary using TF-IDF scoring"""
        text = self._preprocess_text(text)
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                max_features=1000,
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            sentence_scores = {}
            for i in range(len(sentences)):
                # Sum of TF-IDF scores for all terms in the sentence
                score = np.sum(tfidf_matrix[i].toarray())
                
                # Add position bonus
                position_bonus = self._calculate_sentence_position_score(i, len(sentences))
                sentence_scores[i] = score * (1 + position_bonus * 0.2)
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            top_indices = sorted([idx for idx, score in top_sentences[:num_sentences]])
            
            return ' '.join([sentences[i] for i in top_indices])
        
        except Exception as e:
            # Fallback to frequency-based if TF-IDF fails
            print(f"TF-IDF failed, using frequency method: {e}")
            return self.frequency_based_summary(text, num_sentences)
    
    def textrank_summary(self, text, num_sentences=3):
        """Generate summary using TextRank algorithm"""
        text = self._preprocess_text(text)
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
        
        # Apply PageRank algorithm
        try:
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-4)
            
            # Add position weighting
            weighted_scores = {}
            for i in range(len(sentences)):
                position_weight = self._calculate_sentence_position_score(i, len(sentences))
                weighted_scores[i] = scores[i] * (1 + position_weight * 0.3)
            
            # Get top sentences
            ranked_sentences = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
            top_indices = sorted([idx for score, idx in ranked_sentences[:num_sentences]])
            
            return ' '.join([sentences[i] for i in top_indices])
        
        except Exception as e:
            print(f"TextRank failed, using frequency method: {e}")
            return self.frequency_based_summary(text, num_sentences)
    
    def _sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences using multiple metrics"""
        words1 = set(self._tokenize_words(sent1))
        words2 = set(self._tokenize_words(sent2))
        
        if not words1 or not words2:
            return 0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0
        
        # Cosine similarity based on word overlap
        all_words = list(words1.union(words2))
        vec1 = [1 if word in words1 else 0 for word in all_words]
        vec2 = [1 if word in words2 else 0 for word in all_words]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        cosine = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
        
        # Combine similarities
        return (jaccard * 0.6) + (cosine * 0.4)
    

    
    def summarize(self, text, method='tfidf', num_sentences=3):
        """Main summarization method"""
        methods = {
            'frequency': self.frequency_based_summary,
            'tfidf': self.tfidf_summary,
            'textrank': self.textrank_summary
        }
        
        if method not in methods:
            raise ValueError(f"Method must be one of: {list(methods.keys())}")
        
        return methods[method](text, num_sentences)


def demo_summarization():
    """Demonstrate the summarization tool with sample articles"""
    
    sample_articles = [
        {
            "title": "Climate Change and Global Warming",
            "text": """
            Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is a natural phenomenon, scientific evidence shows that human activities have been the main driver of climate change since the 1800s. The burning of fossil fuels like coal, oil, and gas produces greenhouse gas emissions that trap heat in Earth's atmosphere.
            
            The primary greenhouse gases responsible for global warming include carbon dioxide, methane, nitrous oxide, and fluorinated gases. Carbon dioxide levels have increased by over 40% since pre-industrial times, primarily due to fossil fuel use and deforestation. This increase in greenhouse gases has led to a global temperature rise of approximately 1.1 degrees Celsius above pre-industrial levels.
            
            The effects of climate change are already visible worldwide. These include rising sea levels, melting ice caps and glaciers, more frequent extreme weather events, changing precipitation patterns, and shifts in wildlife habitats. Arctic ice is melting at an unprecedented rate, contributing to sea level rise that threatens coastal communities globally.
            
            To address climate change, countries worldwide are implementing various mitigation and adaptation strategies. These include transitioning to renewable energy sources like solar and wind power, improving energy efficiency, protecting and restoring forests, and developing carbon capture technologies. The Paris Agreement, signed by nearly 200 countries, aims to limit global warming to well below 2 degrees Celsius above pre-industrial levels.
            
            Individual actions also play a crucial role in combating climate change. People can reduce their carbon footprint by using public transportation, reducing energy consumption, eating less meat, and supporting sustainable practices. Education and awareness about climate change are essential for building public support for necessary policy changes and lifestyle modifications.
            """
        },
        {
            "title": "Artificial Intelligence in Healthcare",
            "text": """
            Artificial Intelligence is revolutionizing healthcare by providing innovative solutions to longstanding medical challenges. Machine learning algorithms can analyze vast amounts of medical data to identify patterns that human doctors might miss, leading to more accurate diagnoses and personalized treatment plans.
            
            In diagnostic imaging, AI systems have shown remarkable success in detecting diseases such as cancer, diabetic retinopathy, and pneumonia from medical scans. These systems can process thousands of images in minutes, providing rapid screening capabilities that are particularly valuable in areas with limited access to specialist doctors. Deep learning models trained on millions of medical images can now match or exceed the accuracy of experienced radiologists in specific tasks.
            
            Drug discovery is another area where AI is making significant impact. Traditional drug development can take 10-15 years and cost billions of dollars. AI algorithms can analyze molecular structures, predict drug interactions, and identify promising compounds much faster than conventional methods. This acceleration could bring life-saving medications to patients years earlier than traditional approaches.
            
            AI-powered robots are being used in surgery to provide greater precision and minimize invasive procedures. These robotic systems can perform delicate operations with steady hands and enhanced visualization, reducing recovery times and improving patient outcomes. Telemedicine platforms enhanced with AI can provide remote consultations and monitoring, making healthcare more accessible to rural and underserved populations.
            
            However, the integration of AI in healthcare also raises important ethical and regulatory questions. Issues such as data privacy, algorithmic bias, and the need for human oversight must be carefully addressed. As AI continues to evolve, collaboration between technologists, healthcare professionals, and policymakers will be essential to ensure that these powerful tools are used safely and effectively to benefit all patients.
            """
        },
        {
            "title": "The Future of Renewable Energy",
            "text": """
            Renewable energy sources are rapidly becoming the cornerstone of global energy strategy as countries worldwide recognize the urgent need to transition away from fossil fuels. Solar and wind power have experienced dramatic cost reductions over the past decade, making them competitive with traditional energy sources in many markets. The International Energy Agency reports that renewable energy capacity additions reached record levels in recent years, with solar photovoltaic leading the growth.
            
            Technological advancements continue to drive improvements in renewable energy efficiency and storage capabilities. Battery technology has evolved significantly, enabling better energy storage solutions that address the intermittency challenges of solar and wind power. Grid-scale battery installations are becoming more common, allowing utilities to store excess renewable energy during peak production times and release it when demand is high.
            
            Government policies and incentives play a crucial role in accelerating renewable energy adoption. Many countries have implemented feed-in tariffs, tax credits, and renewable portfolio standards to encourage investment in clean energy infrastructure. The European Union has set ambitious targets to achieve carbon neutrality by 2050, while countries like China and India are investing heavily in renewable energy manufacturing and deployment.
            
            The economic benefits of renewable energy extend beyond environmental considerations. The renewable energy sector has become a significant source of employment, creating millions of jobs in manufacturing, installation, and maintenance. Local communities benefit from energy independence and reduced energy costs, while countries can improve their energy security by reducing dependence on imported fossil fuels.
            
            Challenges remain in the transition to renewable energy, including grid integration issues, energy storage costs, and the need for significant infrastructure investments. However, ongoing research and development efforts are addressing these challenges through smart grid technologies, improved energy storage systems, and innovative financing mechanisms. The convergence of renewable energy with digital technologies promises to create more efficient and resilient energy systems for the future.
            """
        }
    ]
    
    # Initialize summarizer
    summarizer = ArticleSummarizer()
    
    print("=" * 80)
    print("ARTICLE SUMMARIZATION TOOL DEMONSTRATION (No NLTK)")
    print("=" * 80)
    
    for article in sample_articles:
        print(f"\n📄 ORIGINAL ARTICLE: {article['title']}")
        print("-" * 60)
        print(article['text'].strip())
        print(f"\nArticle Length: {len(article['text'].split())} words")
        
        print(f"\n📋 SUMMARIES:")
        print("-" * 60)
        
        # Generate summaries using different methods
        methods = ['frequency', 'tfidf', 'textrank']
        
        for method in methods:
            try:
                summary = summarizer.summarize(article['text'], method=method, num_sentences=3)
                word_count = len(summary.split())
                compression_ratio = (word_count / len(article['text'].split())) * 100
                print(f"\n🔹 {method.upper()} Method ({word_count} words, {compression_ratio:.1f}% of original):")
                print(summary)
            except Exception as e:
                print(f"\n🔹 {method.upper()} Method: Error - {str(e)}")
        
        print("\n" + "=" * 80)


def interactive_mode():
    """Interactive mode for custom text summarization"""
    summarizer = ArticleSummarizer()
    
    print("\n🤖 INTERACTIVE SUMMARIZATION MODE")
    print("Enter your text to summarize (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        print("\nEnter text to summarize:")
        user_text = input("> ")
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if len(user_text.strip()) < 100:
            print("Please enter a longer text (at least 100 characters) for meaningful summarization.")
            continue
        
        print("\nChoose summarization method:")
        print("1. Frequency-based")
        print("2. TF-IDF")
        print("3. TextRank")
        
        method_choice = input("Enter choice (1-3) or press Enter for TF-IDF: ").strip()
        
        method_map = {'1': 'frequency', '2': 'tfidf', '3': 'textrank', '': 'tfidf'}
        method = method_map.get(method_choice, 'tfidf')
        
        num_sentences = input("Number of sentences in summary (default 3): ").strip()
        try:
            num_sentences = int(num_sentences) if num_sentences else 3
            num_sentences = max(1, min(num_sentences, 10))  # Limit between 1-10
        except ValueError:
            num_sentences = 3
        
        try:
            summary = summarizer.summarize(user_text, method=method, num_sentences=num_sentences)
            print(f"\n📋 SUMMARY ({method.upper()} method):")
            print("-" * 40)
            print(summary)
            
            # Statistics
            original_words = len(user_text.split())
            summary_words = len(summary.split())
            compression_ratio = (summary_words / original_words) * 100
            
            print(f"\n📊 STATISTICS:")
            print(f"Original: {original_words} words")
            print(f"Summary: {summary_words} words")
            print(f"Compression: {compression_ratio:.1f}%")
            print(f"Reduction: {100 - compression_ratio:.1f}%")
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")


def analyze_text_statistics(text):
    """Analyze and display text statistics"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    print(f"📊 TEXT ANALYSIS:")
    print(f"Total characters: {len(text)}")
    print(f"Total words: {len(words)}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Average words per sentence: {len(words)/len(sentences):.1f}")
    print(f"Average characters per word: {sum(len(w) for w in words)/len(words):.1f}")


if __name__ == "__main__":
    print("Article Summarization Tool (No NLTK Required)")
    print("=" * 45)
    
    # Run demonstration
    demo_summarization()
    
    # Ask user if they want interactive mode
    print("\nWould you like to try the interactive mode? (y/n): ", end="")
    try:
        choice = input().lower()
        if choice in ['y', 'yes']:
            interactive_mode()
        else:
            print("\nThank you for using the Article Summarization Tool!")
    except KeyboardInterrupt:
        print("\n\nThank you for using the Article Summarization Tool!")