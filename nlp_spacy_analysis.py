import spacy
from spacy import displacy
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load spaCy model
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install the English model: python -m spacy download en_core_web_sm")
    exit()

# Sample Amazon product reviews (in practice, you'd load from a dataset)
sample_reviews = [
    "I love my new iPhone 13 Pro Max! The camera quality is amazing and the battery life is excellent. Apple really outdid themselves this time.",
    "The Samsung Galaxy S22 is disappointing. The screen has issues and Samsung's customer service was unhelpful. Would not recommend.",
    "Amazon delivery was fast as always. The Nike Air Max shoes fit perfectly and the quality is outstanding. Nike continues to impress.",
    "Sony WH-1000XM4 headphones are the best I've ever owned. The noise cancellation from Sony is incredible and the sound quality is perfect.",
    "The MacBook Pro from Apple is overpriced for what you get. Better alternatives exist. Apple's pricing strategy is questionable.",
    "Microsoft Surface Pro 8 is a great tablet-laptop hybrid. Microsoft has really improved the design and performance significantly.",
    "Google Pixel 6 Pro camera is revolutionary. Google's computational photography is unmatched. The phone exceeded my expectations completely.",
    "Tesla Model 3 is an incredible electric vehicle. Tesla's innovation in the automotive industry is remarkable. Highly recommended purchase."
]

print(f"Analyzing {len(sample_reviews)} sample reviews...")

def extract_entities_and_sentiment(reviews):
    """Extract named entities and perform sentiment analysis on reviews."""
    results = []
    
    for i, review in enumerate(reviews):
        print(f"\nProcessing review {i+1}...")
        
        # Process with spaCy
        doc = nlp(review)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        
        # Extract product names and brands (focusing on ORG and PRODUCT entities)
        brands = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'PERSON']]
        
        # Rule-based sentiment analysis using TextBlob
        blob = TextBlob(review)
        sentiment_score = blob.sentiment.polarity
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment = 'Positive'
        elif sentiment_score < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        results.append({
            'review_id': i+1,
            'review_text': review,
            'entities': entities,
            'brands': brands,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment
        })
        
        print(f"Entities found: {len(entities)}")
        print(f"Brands/Products: {brands}")
        print(f"Sentiment: {sentiment} (Score: {sentiment_score:.3f})")
    
    return results

# Perform analysis
analysis_results = extract_entities_and_sentiment(sample_reviews)

# Display detailed results
print("\n" + "="*50)
print("DETAILED ANALYSIS RESULTS")
print("="*50)

for result in analysis_results:
    print(f"\nReview {result['review_id']}:")
    print(f"Text: {result['review_text'][:100]}...")
    print(f"Sentiment: {result['sentiment']} ({result['sentiment_score']:.3f})")
    print(f"Brands/Products found: {', '.join(result['brands']) if result['brands'] else 'None'}")
    
    if result['entities']:
        print("Named Entities:")
        for ent in result['entities']:
            print(f"  - {ent['text']} ({ent['label']}: {ent['description']})")

# Aggregate analysis
print("\n" + "="*50)
print("AGGREGATE ANALYSIS")
print("="*50)

# Sentiment distribution
sentiments = [r['sentiment'] for r in analysis_results]
sentiment_counts = Counter(sentiments)
print(f"\nSentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment}: {count} reviews ({count/len(analysis_results)*100:.1f}%)")

# Brand/Product mentions
all_brands = []
for result in analysis_results:
    all_brands.extend(result['brands'])

brand_counts = Counter(all_brands)
print(f"\nTop Brands/Products Mentioned:")
for brand, count in brand_counts.most_common():
    print(f"  {brand}: {count} mentions")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sentiment distribution pie chart
axes[0, 0].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
axes[0, 0].set_title('Sentiment Distribution')

# Sentiment scores histogram
sentiment_scores = [r['sentiment_score'] for r in analysis_results]
axes[0, 1].hist(sentiment_scores, bins=10, alpha=0.7, color='skyblue')
axes[0, 1].set_title('Sentiment Score Distribution')
axes[0, 1].set_xlabel('Sentiment Score')
axes[0, 1].set_ylabel('Frequency')

# Brand mentions bar chart
if brand_counts:
    brands, counts = zip(*brand_counts.most_common())
    axes[1, 0].bar(brands, counts)
    axes[1, 0].set_title('Brand/Product Mentions')
    axes[1, 0].set_xlabel('Brand/Product')
    axes[1, 0].set_ylabel('Mentions')
    axes[1, 0].tick_params(axis='x', rotation=45)

# Entity types distribution
all_entity_types = []
for result in analysis_results:
    for ent in result['entities']:
        all_entity_types.append(ent['label'])

entity_type_counts = Counter(all_entity_types)
if entity_type_counts:
    types, type_counts = zip(*entity_type_counts.most_common())
    axes[1, 1].bar(types, type_counts)
    axes[1, 1].set_title('Entity Types Distribution')
    axes[1, 1].set_xlabel('Entity Type')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Export results to DataFrame for further analysis
df_results = pd.DataFrame([
    {
        'review_id': r['review_id'],
        'review_text': r['review_text'],
        'sentiment': r['sentiment'],
        'sentiment_score': r['sentiment_score'],
        'brands_found': ', '.join(r['brands']),
        'entity_count': len(r['entities'])
    }
    for r in analysis_results
])

print("\nResults DataFrame:")
print(df_results)

print("\n=== TASK 3 COMPLETED SUCCESSFULLY ===")