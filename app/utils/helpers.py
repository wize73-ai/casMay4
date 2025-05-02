
"""
Helpers Module for CasaLingua
Provides utility functions for the RAG system and language learning components
"""

__all__ = [
    "detect_language", "normalize_text", "split_into_sentences", "tokenize_text",
    "calculate_text_similarity", "extract_keywords", "get_stopwords", "read_json_file",
    "write_json_file", "create_unique_id", "load_embeddings_from_file", "save_embeddings_to_file",
    "batch_generator", "cosine_similarity", "euclidean_distance", "find_nearest_neighbors",
    "normalize_vectors", "reduce_dimensions", "Timer", "profile_function", "detect_difficulty_level",
    "get_cognates", "conjugate_verb", "get_language_name", "get_spaced_repetition_interval"
]

import os
import json
import time
import logging
import hashlib
import re
import string
import unicodedata
import difflib
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set, Generator
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# --------------------
# Tokenization and Language Utilities
# --------------------

def detect_language(text: str) -> str:
    """
    Detect language of text using common word frequencies
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected language code ('en', 'es', 'fr', etc.) or 'unknown'
    """
    # Common words in different languages
    language_markers = {
        'en': ['the', 'and', 'of', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on', 'are', 'with', 'as'],
        'es': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'por', 'con', 'su', 'para', 'como'],
        'fr': ['le', 'la', 'de', 'et', 'est', 'en', 'que', 'un', 'une', 'du', 'dans', 'qui', 'il', 'à', 'ce', 'pas'],
        'de': ['der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'den', 'mit', 'nicht', 'ein', 'für', 'von', 'sie', 'auf', 'dem'],
        'it': ['il', 'la', 'di', 'e', 'che', 'un', 'a', 'per', 'in', 'sono', 'mi', 'con', 'si', 'ho', 'lo', 'non']
    }
    
    # Prepare text for analysis
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Count word frequencies for each language
    scores = {lang: 0 for lang in language_markers}
    word_count = len(words)
    
    if word_count == 0:
        return 'unknown'
        
    for lang, markers in language_markers.items():
        lang_words = set(markers)
        matches = sum(1 for word in words if word in lang_words)
        scores[lang] = matches / word_count
    
    # Get language with highest score above threshold
    best_lang = max(scores.items(), key=lambda x: x[1])
    
    if best_lang[1] >= 0.04:  # At least 4% of words match language markers
        return best_lang[0]
    
    return 'unknown'


def normalize_text(text: str, lowercase: bool = True, remove_accents: bool = False) -> str:
    """
    Normalize text for processing
    
    Args:
        text (str): Text to normalize
        lowercase (bool): Whether to convert to lowercase
        remove_accents (bool): Whether to remove accents
        
    Returns:
        str: Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove accents if requested
    if remove_accents:
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
    
    return text


def split_into_sentences(text: str, language: str = 'en') -> List[str]:
    """
    Split text into sentences with language awareness
    
    Args:
        text (str): Text to split
        language (str): Language code for better splitting
        
    Returns:
        List[str]: List of sentences
    """
    try:
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text, language=language)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback to simple regex splitting
        sentence_endings = r'[.!?][\s\n]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]


def tokenize_text(text: str, language: str = 'en') -> List[str]:
    """
    Tokenize text into words
    
    Args:
        text (str): Text to tokenize
        language (str): Language code for better tokenization
        
    Returns:
        List[str]: List of tokens
    """
    try:
        # Use NLTK's word tokenizer
        return word_tokenize(text, language=language)
    except Exception:
        # Fallback to simple regex tokenization
        return re.findall(r'\b\w+\b', text)


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    # Normalize texts
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Use difflib sequence matcher
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def extract_keywords(text: str, top_n: int = 10, language: str = 'en') -> List[str]:
    """
    Extract important keywords from text
    
    Args:
        text (str): Text to analyze
        top_n (int): Number of keywords to extract
        language (str): Language code
        
    Returns:
        List[str]: List of keywords
    """
    # Normalize text
    text = normalize_text(text)
    
    # Tokenize
    tokens = tokenize_text(text, language)
    
    # Remove stopwords
    stopwords = get_stopwords(language)
    tokens = [token for token in tokens if token.lower() not in stopwords]
    
    # Count frequencies
    word_freq = {}
    for token in tokens:
        if len(token) >= 3:  # Skip very short words
            word_freq[token] = word_freq.get(token, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [word for word, freq in sorted_words[:top_n]]


def get_stopwords(language: str) -> Set[str]:
    """
    Get stopwords for a language
    
    Args:
        language (str): Language code
        
    Returns:
        Set[str]: Set of stopwords
    """
    # Common stopwords for major languages
    stopwords = {
        'en': {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'},
        'es': {'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella', 'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es', 'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban', 'estabas', 'estad', 'estada', 'estadas', 'estado', 'estados', 'estamos', 'estando', 'estar', 'estaremos', 'estará', 'estarán', 'estarás', 'estaré', 'estaréis', 'estaría', 'estaríais', 'estaríamos', 'estarían', 'estarías', 'estas', 'este', 'estemos', 'esto', 'estos', 'estoy', 'estuve', 'estuviera', 'estuvierais', 'estuvieran', 'estuvieras', 'estuvieron', 'estuviese', 'estuvieseis', 'estuviesen', 'estuvieses', 'estuvimos', 'estuviste', 'estuvisteis', 'estuviéramos', 'estuviésemos', 'estuvo', 'está', 'estábamos', 'estáis', 'están', 'estás', 'esté', 'estéis', 'estén', 'estés', 'fue', 'fuera', 'fuerais', 'fueran', 'fueras', 'fueron', 'fuese', 'fueseis', 'fuesen', 'fueses', 'fui', 'fuimos', 'fuiste', 'fuisteis', 'fuéramos', 'fuésemos', 'ha', 'habida', 'habidas', 'habido', 'habidos', 'habiendo', 'habremos', 'habrá', 'habrán', 'habrás', 'habré', 'habréis', 'habría', 'habríais', 'habríamos', 'habrían', 'habrías', 'habéis', 'había', 'habíais', 'habíamos', 'habían', 'habías', 'han', 'has', 'hasta', 'hay', 'haya', 'hayamos', 'hayan', 'hayas', 'hayáis', 'he', 'hemos', 'hube', 'hubiera', 'hubierais', 'hubieran', 'hubieras', 'hubieron', 'hubiese', 'hubieseis', 'hubiesen', 'hubieses', 'hubimos', 'hubiste', 'hubisteis', 'hubiéramos', 'hubiésemos', 'hubo', 'la', 'las', 'le', 'les', 'lo', 'los', 'me', 'mi', 'mis', 'mucho', 'muchos', 'muy', 'más', 'mí', 'mía', 'mías', 'mío', 'míos', 'nada', 'ni', 'no', 'nos', 'nosotras', 'nosotros', 'nuestra', 'nuestras', 'nuestro', 'nuestros', 'o', 'os', 'otra', 'otras', 'otro', 'otros', 'para', 'pero', 'poco', 'por', 'porque', 'que', 'quien', 'quienes', 'qué', 'se', 'sea', 'seamos', 'sean', 'seas', 'seremos', 'será', 'serán', 'serás', 'seré', 'seréis', 'sería', 'seríais', 'seríamos', 'serían', 'serías', 'seáis', 'si', 'sido', 'siendo', 'sin', 'sobre', 'sois', 'somos', 'son', 'soy', 'su', 'sus', 'suya', 'suyas', 'suyo', 'suyos', 'sí', 'también', 'tanto', 'te', 'tendremos', 'tendrá', 'tendrán', 'tendrás', 'tendré', 'tendréis', 'tendría', 'tendríais', 'tendríamos', 'tendrían', 'tendrías', 'tened', 'tenemos', 'tenga', 'tengamos', 'tengan', 'tengas', 'tengo', 'tengáis', 'tenida', 'tenidas', 'tenido', 'tenidos', 'teniendo', 'tenéis', 'tenía', 'teníais', 'teníamos', 'tenían', 'tenías', 'ti', 'tiene', 'tienen', 'tienes', 'todo', 'todos', 'tu', 'tus', 'tuve', 'tuviera', 'tuvierais', 'tuvieran', 'tuvieras', 'tuvieron', 'tuviese', 'tuvieseis', 'tuviesen', 'tuvieses', 'tuvimos', 'tuviste', 'tuvisteis', 'tuviéramos', 'tuviésemos', 'tuvo', 'tuya', 'tuyas', 'tuyo', 'tuyos', 'tú', 'un', 'una', 'uno', 'unos', 'vosotras', 'vosotros', 'vuestra', 'vuestras', 'vuestro', 'vuestros', 'y', 'ya', 'yo', 'él', 'éramos'},
        'fr': {'a', 'à', 'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ni', 'notre', 'nous', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'si', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'n', 'm', 's', 't', 'y'},
        'de': {'ab', 'aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere', 'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das', 'daß', 'dass', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'können', 'könnte', 'machen', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst', 'um', 'und', 'uns', 'unse', 'unsem', 'unsen', 'unser', 'unses', 'unter', 'viel', 'vom', 'von', 'vor', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'während', 'würde', 'würden', 'zu', 'zum', 'zur', 'zwar', 'zwischen', 'über'}
    }
    
    return stopwords.get(language, set())


# --------------------
# File and Data Handling Utilities
# --------------------

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        Dict[str, Any]: JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return {}


def write_json_file(file_path: str, data: Any, indent: int = 2) -> bool:
    """
    Write data to a JSON file
    
    Args:
        file_path (str): Path to JSON file
        data (Any): Data to write
        indent (int): JSON indentation
        
    Returns:
        bool: Success status
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False


def create_unique_id(text: str, prefix: str = '') -> str:
    """
    Create a unique ID for a text
    
    Args:
        text (str): Text to hash
        prefix (str): Optional prefix for the ID
        
    Returns:
        str: Unique ID
    """
    # Create MD5 hash
    hasher = hashlib.md5()
    hasher.update(text.encode('utf-8'))
    hash_id = hasher.hexdigest()
    
    # Add prefix if provided
    if prefix:
        return f"{prefix}_{hash_id}"
    
    return hash_id


def load_embeddings_from_file(file_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load text embeddings from a file
    
    Args:
        file_path (str): Path to embeddings file
        
    Returns:
        Tuple[List[str], np.ndarray]: Texts and embeddings
    """
    try:
        data = read_json_file(file_path)
        texts = data.get('texts', [])
        embeddings = np.array(data.get('embeddings', []))
        return texts, embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {file_path}: {e}")
        return [], np.array([])


def save_embeddings_to_file(file_path: str, texts: List[str], embeddings: np.ndarray) -> bool:
    """
    Save text embeddings to a file
    
    Args:
        file_path (str): Path to save embeddings
        texts (List[str]): List of texts
        embeddings (np.ndarray): Text embeddings
        
    Returns:
        bool: Success status
    """
    try:
        data = {
            'texts': texts,
            'embeddings': embeddings.tolist()
        }
        return write_json_file(file_path, data)
    except Exception as e:
        logger.error(f"Error saving embeddings to {file_path}: {e}")
        return False


def batch_generator(items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Generate batches from a list of items
    
    Args:
        items (List[Any]): Items to batch
        batch_size (int): Size of each batch
        
    Yields:
        List[Any]: Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# --------------------
# Embedding and Vector Utilities
# --------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        v1 (np.ndarray): First vector
        v2 (np.ndarray): Second vector
        
    Returns:
        float: Cosine similarity (-1 to 1)
    """
    if v1.size == 0 or v2.size == 0:
        return 0.0
        
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors
    
    Args:
        v1 (np.ndarray): First vector
        v2 (np.ndarray): Second vector
        
    Returns:
        float: Euclidean distance
    """
    if v1.size == 0 or v2.size == 0:
        return float('inf')
        
    return np.linalg.norm(v1 - v2)


def find_nearest_neighbors(query_vector: np.ndarray, 
                         vectors: np.ndarray, 
                         k: int = 5,
                         method: str = 'cosine') -> List[Tuple[int, float]]:
    """
    Find nearest neighbors of a query vector
    
    Args:
        query_vector (np.ndarray): Query vector
        vectors (np.ndarray): Matrix of vectors to search
        k (int): Number of neighbors to return
        method (str): 'cosine' or 'euclidean'
        
    Returns:
        List[Tuple[int, float]]: List of (index, similarity) tuples
    """
    if vectors.size == 0:
        return []
        
    # Calculate similarities or distances
    if method == 'cosine':
        similarities = np.array([cosine_similarity(query_vector, v) for v in vectors])
        indices = np.argsort(-similarities)  # Sort by descending similarity
        scores = similarities[indices]
    else:  # euclidean
        distances = np.array([euclidean_distance(query_vector, v) for v in vectors])
        indices = np.argsort(distances)  # Sort by ascending distance
        scores = -distances[indices]  # Negate for consistent comparison
    
    # Return top k results
    k = min(k, len(indices))
    return [(int(indices[i]), float(scores[i])) for i in range(k)]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length
    
    Args:
        vectors (np.ndarray): Vectors to normalize
        
    Returns:
        np.ndarray: Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Handle zero norms
    norms[norms == 0] = 1.0
    return vectors / norms


def reduce_dimensions(vectors: np.ndarray, dimensions: int = 50) -> np.ndarray:
    """
    Reduce dimensionality of vectors using SVD
    
    Args:
        vectors (np.ndarray): Vectors to reduce
        dimensions (int): Target dimensions
        
    Returns:
        np.ndarray: Reduced vectors
    """
    from sklearn.decomposition import TruncatedSVD
    
    # Skip if already lower dimensional
    if vectors.shape[1] <= dimensions:
        return vectors
        
    try:
        svd = TruncatedSVD(n_components=dimensions)
        return svd.fit_transform(vectors)
    except Exception as e:
        logger.error(f"Error reducing dimensions: {e}")
        return vectors


# --------------------
# Performance Measurement Utilities
# --------------------

class Timer:
    """Utility class for timing operations"""
    
    def __init__(self, name: str = 'Operation'):
        """
        Initialize timer
        
        Args:
            name (str): Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer when entering context"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log elapsed time when exiting context"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {elapsed:.4f} seconds")
    
    def elapsed(self) -> float:
        """
        Get elapsed time
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
            
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


def profile_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to profile function execution time
    
    Args:
        func (Callable): Function to profile
        
    Returns:
        Callable: Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Function {func.__name__} completed in {elapsed:.4f} seconds")
        return result
    
    return wrapper


# --------------------
# Language Learning Specific Utilities
# --------------------

def detect_difficulty_level(text: str, language: str = 'en') -> str:
    """
    Estimate difficulty level of a text for language learners
    
    Args:
        text (str): Text to analyze
        language (str): Language code
        
    Returns:
        str: Difficulty level ('beginner', 'intermediate', 'advanced')
    """
    # Tokenize text
    words = tokenize_text(text, language)
    
    if not words:
        return 'beginner'
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Calculate unique word ratio
    unique_word_ratio = len(set(word.lower() for word in words)) / len(words)
    
    # Calculate sentence length
    sentences = split_into_sentences(text, language)
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    # Combine metrics for difficulty score
    difficulty_score = (
        (avg_word_length * 0.3) + 
        (unique_word_ratio * 30) + 
        (avg_sentence_length * 0.2)
    )
    
    # Classify based on score
    if difficulty_score < 7:
        return 'beginner'
    elif difficulty_score < 10:
        return 'intermediate'
    else:
        return 'advanced'


def get_cognates(word: str, source_lang: str, target_lang: str) -> List[str]:
    """
    Find potential cognates (similar words) in target language
    
    Args:
        word (str): Word to find cognates for
        source_lang (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        List[str]: Potential cognates
    """
    # Example cognate patterns (very simplified)
    # In a real implementation, this would use a database or API
    cognate_patterns = {
        ('en', 'es'): [
            (r'tion$', 'ción'),
            (r'ty$', 'dad'),
            (r'ic$', 'ico'),
            (r'ist$', 'ista'),
            (r'ism$', 'ismo'),
            (r'ment$', 'mento'),
            (r'or$', 'or')
        ],
        ('en', 'fr'): [
            (r'tion$', 'tion'),
            (r'ty$', 'té'),
            (r'ic$', 'ique'),
            (r'ist$', 'iste'),
            (r'ism$', 'isme'),
            (r'or$', 'eur')
        ]
    }
    
    # Check if we have patterns for this language pair
    key = (source_lang, target_lang)
    if key not in cognate_patterns:
        return []
    
    # Apply patterns
    potential_cognates = []
    for pattern, replacement in cognate_patterns[key]:
        if re.search(pattern, word, re.IGNORECASE):
            cognate = re.sub(pattern, replacement, word, flags=re.IGNORECASE)
            potential_cognates.append(cognate)
    
    return potential_cognates


def conjugate_verb(verb: str, tense: str, subject: str, language: str) -> str:
    """
    Conjugate a verb in the given language
    
    Args:
        verb (str): Verb in infinitive form
        tense (str): Tense (present, past, future)
        subject (str): Subject (I, you, he/she/it, we, you all, they)
        language (str): Language code
        
    Returns:
        str: Conjugated verb
    """
    # This is a simplified example - real implementation would use a more comprehensive system
    # Spanish verb conjugation example
    if language == 'es':
        # Check if regular -ar verb
        if verb.endswith('ar'):
            stem = verb[:-2]
            if tense == 'present':
                if subject == 'I':
                    return f"{stem}o"
                elif subject == 'you':
                    return f"{stem}as"
                elif subject == 'he/she/it':
                    return f"{stem}a"
                elif subject == 'we':
                    return f"{stem}amos"
                elif subject == 'you all':
                    return f"{stem}áis"
                elif subject == 'they':
                    return f"{stem}an"
    
    # Return original if conjugation not implemented
    return verb


def get_language_name(code: str) -> str:
    """
    Get full language name from language code
    
    Args:
        code (str): Language code (e.g., 'en', 'es')
        
    Returns:
        str: Language name
    """
    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'zh': 'Chinese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'pl': 'Polish',
        'vi': 'Vietnamese',
        'th': 'Thai'
    }
    
    return language_names.get(code.lower(), 'Unknown')


def get_spaced_repetition_interval(level: int, success: bool) -> int:
    """
    Calculate spaced repetition interval for vocabulary
    
    Args:
        level (int): Current knowledge level (0-5)
        success (bool): Whether recall was successful
        
    Returns:
        int: Interval in days
    """
    if not success:
        return 1  # Reset to 1 day on failure
    
    # Calculate interval based on level
    if level == 0:
        return 1
    elif level == 1:
        return 2
    elif level == 2:
        return 4
    elif level == 3:
        return 7
    elif level == 4:
        return 15
    else:  # level >= 5
        return 30


# Example usage
if __name__ == "__main__":
    # Example text
    text = "Hello, this is an example text for demonstrating helper functions in CasaLingua."
    
    # Detect language
    detected_lang = detect_language(text)
    print(f"Detected language: {detected_lang}")
    
    # Tokenize
    tokens = tokenize_text(text)
    print(f"Tokens: {tokens}")
    
    # Extract keywords
    keywords = extract_keywords(text)
    print(f"Keywords: {keywords}")
    
    # Time an operation
    with Timer("Example operation"):
        # Simulate work
        time.sleep(1)
    
    # Create a unique ID
    unique_id = create_unique_id(text)
    print(f"Unique ID: {unique_id}")
    
    # Detect difficulty level
    difficulty = detect_difficulty_level(text)
    print(f"Difficulty level: {difficulty}")