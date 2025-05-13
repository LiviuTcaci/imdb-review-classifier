import re
from transformers import DistilBertTokenizer

def clean_text(text):
    """
    Clean IMDb review text with fixed preprocessing steps:
    1. Remove HTML tags
    2. Convert to lowercase
    3. Keep alphanumeric characters, basic punctuation and spaces
    4. Remove extra whitespace
    5. Strip leading/trailing whitespace
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Keep alphanumeric characters, basic punctuation and spaces
    text = re.sub(r'[^a-z0-9\s\.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def tokenize_text(text, max_length=512, truncation=True, padding=True):
    """
    Tokenize text using DistilBERT tokenizer with configurable options.
    
    Args:
        text (str): Input text to tokenize
        max_length (int): Maximum sequence length (default: 512)
        truncation (bool): Whether to truncate sequences (default: True)
        padding (bool): Whether to pad sequences (default: True)
        
    Returns:
        dict: Tokenized text with input_ids and attention_mask as lists
    """
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Tokenize text with fixed max_length of 512
    tokenized = tokenizer(
        text,
        max_length=512,  # Fixed value based on EDA analysis
        truncation=True,  # Always truncate to max_length
        padding='max_length',  # Pad to max_length
        return_tensors=None  # Return lists instead of tensors
    )
    return tokenized

