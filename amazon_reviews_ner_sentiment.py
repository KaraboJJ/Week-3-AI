import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

reviews = [
    "I love the Samsung Galaxy phone, it has amazing battery life.",
    "The Apple MacBook is overpriced and has poor performance.",
    "Sony headphones are really comfortable and sound great!",
    "I was disappointed by the quality of the Nike running shoes.",
    "The new Dell laptop exceeded my expectations."
]

brands = ["Samsung", "Apple", "Sony", "Nike", "Dell"]
products = ["Galaxy", "MacBook", "headphones", "running shoes", "laptop"]

brand_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
brand_patterns = [nlp.make_doc(text) for text in brands]
brand_matcher.add("BRAND", brand_patterns)

product_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
product_patterns = [nlp.make_doc(text) for text in products]
product_matcher.add("PRODUCT", product_patterns)

positive_words = ["love", "amazing", "comfortable", "great", "exceeded"]
negative_words = ["disappointed", "poor", "overpriced"]

def analyze_review(text):
    doc = nlp(text)
    
    brands_found = [doc[start:end].text for match_id, start, end in brand_matcher(doc)]
    products_found = [doc[start:end].text for match_id, start, end in product_matcher(doc)]
    
    text_lower = text.lower()
    sentiment = "Neutral"
    if any(word in text_lower for word in positive_words):
        sentiment = "Positive"
    if any(word in text_lower for word in negative_words):
        sentiment = "Negative"
    
    return {
        "text": text,
        "brands": brands_found,
        "products": products_found,
        "sentiment": sentiment
    }

for review in reviews:
    result = analyze_review(review)
    print(f"Review: {result['text']}")
    print(f"Brands found: {result['brands']}")
    print(f"Products found: {result['products']}")
    print(f"Sentiment: {result['sentiment']}")
    print("-" * 50)
