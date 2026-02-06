import pandas as pd
import os

# Tạo dữ liệu mẫu để demo
sample_data = {
    'review': [
        "This movie is absolutely fantastic! The acting was superb and the plot was engaging.",
        "I loved every minute of this film. Great cinematography and excellent performances.",
        "One of the best movies I've ever seen. Highly recommend it to everyone!",
        "Amazing storyline with incredible visual effects. A masterpiece!",
        "Brilliant acting and direction. This movie exceeded all my expectations.",
        "Wonderful film with great character development and emotional depth.",
        "Excellent movie with outstanding performances from all actors.",
        "This film is a work of art. Beautiful and moving story.",
        "Great movie with perfect pacing and amazing soundtrack.",
        "Loved the plot twists and the incredible ending. Must watch!",
        
        "This movie was terrible. Poor acting and boring plot.",
        "I hated this film. Complete waste of time and money.",
        "One of the worst movies ever made. Avoid at all costs!",
        "Awful storyline with terrible visual effects. A disaster!",
        "Bad acting and poor direction. This movie was disappointing.",
        "Horrible film with weak character development and no depth.",
        "Terrible movie with awful performances from all actors.",
        "This film is garbage. Boring and pointless story.",
        "Bad movie with poor pacing and annoying soundtrack.",
        "Hated the confusing plot and the terrible ending. Skip this!",
        
        # Thêm nhiều mẫu hơn
        "The movie was okay, nothing special but watchable.",
        "Decent film with some good moments and decent acting.",
        "Not bad, but could have been better. Average movie.",
        "It's an alright movie, not great but not terrible either.",
        "The film has its moments but overall just mediocre.",
        
        # Positive samples
        "Incredible movie with stunning visuals and great story!",
        "Perfect film! Everything about it was amazing.",
        "Loved the characters and the emotional journey.",
        "Outstanding cinematography and brilliant performances.",
        "This movie touched my heart. Absolutely beautiful.",
        "Fantastic direction and incredible acting throughout.",
        "Amazing film that kept me engaged from start to finish.",
        "Brilliant storytelling with perfect execution.",
        "Wonderful movie with great message and superb acting.",
        "Excellent film that deserves all the praise it gets.",
        
        # Negative samples  
        "Boring movie with predictable plot and bad acting.",
        "Waste of time. Nothing interesting happens in this film.",
        "Poor script and terrible direction ruined this movie.",
        "Disappointing film with weak storyline and bad effects.",
        "Awful movie that made no sense. Very confusing.",
        "Bad film with annoying characters and poor dialogue.",
        "Terrible movie that was painful to watch.",
        "Horrible acting and boring story made this unwatchable.",
        "Poor quality film with no redeeming qualities.",
        "Bad movie that fails on every level."
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative'
    ]
}

# Tạo thư mục nếu chưa có
os.makedirs("data/raw", exist_ok=True)

# Tạo DataFrame và lưu
df = pd.DataFrame(sample_data)
df.to_csv("data/raw/dataset.csv", index=False)

print(f"Đã tạo dataset mẫu với {len(df)} samples:")
print(f"- Positive: {len(df[df['sentiment'] == 'positive'])}")
print(f"- Negative: {len(df[df['sentiment'] == 'negative'])}")
print("\nMẫu dữ liệu:")
print(df.head())