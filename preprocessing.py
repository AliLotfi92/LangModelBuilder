import re
from langdetect import detect

#replacing profanities
def clean_text(text, profanities):
    pattern = r'\b(' + '|'.join(map(re.escape, profanities)) + r')\b'
    cleaned_text = re.sub(pattern, '[PROFANITY]', text, flags=re.IGNORECASE)
    return cleaned_text

#remove non-english documents/text
def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception as e:
        print(f"Language detection error: {e}")
        return False
