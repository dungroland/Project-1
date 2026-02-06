import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Thiết lập style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📊 PHÂN TÍCH TRỌNG SỐ TF-IDF")
print("=" * 60)

# 1. Load dữ liệu và vectorizer
print("\n🔄 Đang tải dữ liệu và vectorizer...")
train_df = pd.read_csv("data/processed/train.csv")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# 2. Transform dữ liệu
X_train_tfidf = tfidf.transform(train_df['review'].values.astype('U'))

# 3. Tính trọng số TF-IDF trung bình cho mỗi từ
print("🔄 Đang tính trọng số TF-IDF trung bình...")
feature_names = tfidf.get_feature_names_out()
tfidf_means = np.asarray(X_train_tfidf.mean(axis=0)).ravel()

# 4. Tạo DataFrame để phân tích
tfidf_df = pd.DataFrame({
    'word': feature_names,
    'tfidf_mean': tfidf_means
}).sort_values('tfidf_mean', ascending=False)

# 5. In top 20 từ có TF-IDF cao nhất
print("\n🔝 TOP 20 TỪ CÓ TRỌNG SỐ TF-IDF CAO NHẤT:")
print("-" * 60)
for idx, row in tfidf_df.head(20).iterrows():
    print(f"{row['word']:<25} {row['tfidf_mean']:.6f}")

# 6. Phân tích theo loại từ
unigrams = tfidf_df[tfidf_df['word'].str.split().str.len() == 1]
bigrams = tfidf_df[tfidf_df['word'].str.split().str.len() == 2]

print(f"\n📈 THỐNG KÊ:")
print(f"• Tổng số từ: {len(tfidf_df):,}")
print(f"• Unigrams: {len(unigrams):,}")
print(f"• Bigrams: {len(bigrams):,}")
print(f"• TF-IDF trung bình: {tfidf_means.mean():.6f}")
print(f"• TF-IDF max: {tfidf_means.max():.6f}")
print(f"• TF-IDF min: {tfidf_means.min():.6f}")

# 7. Tạo biểu đồ
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Phân Tích Trọng Số TF-IDF', fontsize=16, fontweight='bold')

# 7.1. Top 20 từ có TF-IDF cao nhất
top_20 = tfidf_df.head(20)
axes[0,0].barh(range(len(top_20)), top_20['tfidf_mean'], color='steelblue', alpha=0.8, edgecolor='black')
axes[0,0].set_yticks(range(len(top_20)))
axes[0,0].set_yticklabels(top_20['word'])
axes[0,0].invert_yaxis()
axes[0,0].set_xlabel('TF-IDF Mean Score')
axes[0,0].set_title('Top 20 Từ Có TF-IDF Cao Nhất', fontweight='bold')
axes[0,0].grid(axis='x', alpha=0.3)

# Thêm giá trị vào thanh
for i, (idx, row) in enumerate(top_20.iterrows()):
    axes[0,0].text(row['tfidf_mean'], i, f" {row['tfidf_mean']:.4f}", 
                   va='center', fontsize=9, fontweight='bold')

# 7.2. Top 15 Unigrams
top_unigrams = unigrams.head(15)
axes[0,1].barh(range(len(top_unigrams)), top_unigrams['tfidf_mean'], 
               color='green', alpha=0.7, edgecolor='black')
axes[0,1].set_yticks(range(len(top_unigrams)))
axes[0,1].set_yticklabels(top_unigrams['word'])
axes[0,1].invert_yaxis()
axes[0,1].set_xlabel('TF-IDF Mean Score')
axes[0,1].set_title('Top 15 Unigrams (1 từ)', fontweight='bold')
axes[0,1].grid(axis='x', alpha=0.3)

# 7.3. Top 15 Bigrams
top_bigrams = bigrams.head(15)
axes[1,0].barh(range(len(top_bigrams)), top_bigrams['tfidf_mean'], 
               color='orange', alpha=0.7, edgecolor='black')
axes[1,0].set_yticks(range(len(top_bigrams)))
axes[1,0].set_yticklabels(top_bigrams['word'])
axes[1,0].invert_yaxis()
axes[1,0].set_xlabel('TF-IDF Mean Score')
axes[1,0].set_title('Top 15 Bigrams (2 từ)', fontweight='bold')
axes[1,0].grid(axis='x', alpha=0.3)

# 7.4. Phân bố TF-IDF
axes[1,1].hist(tfidf_means, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1,1].axvline(tfidf_means.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tfidf_means.mean():.4f}')
axes[1,1].axvline(np.median(tfidf_means), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(tfidf_means):.4f}')
axes[1,1].set_xlabel('TF-IDF Score')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Phân Bố Trọng Số TF-IDF', fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('tfidf_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Biểu đồ đã được lưu vào 'tfidf_analysis.png'")

# 8. Tạo biểu đồ riêng cho Top 30 từ (dễ nhìn hơn)
fig2, ax = plt.subplots(figsize=(12, 10))
top_30 = tfidf_df.head(30)

bars = ax.barh(range(len(top_30)), top_30['tfidf_mean'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(top_30))), 
               alpha=0.8, edgecolor='black')

ax.set_yticks(range(len(top_30)))
ax.set_yticklabels(top_30['word'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('TF-IDF Mean Score', fontsize=12, fontweight='bold')
ax.set_title('Top 30 Từ Có Trọng Số TF-IDF Cao Nhất', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Thêm giá trị vào thanh
for i, (idx, row) in enumerate(top_30.iterrows()):
    ax.text(row['tfidf_mean'], i, f" {row['tfidf_mean']:.5f}", 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('tfidf_top30.png', dpi=300, bbox_inches='tight')
print("✅ Biểu đồ Top 30 đã được lưu vào 'tfidf_top30.png'")

# 9. Lưu kết quả ra CSV
tfidf_df.to_csv('tfidf_weights.csv', index=False)
print("✅ Dữ liệu TF-IDF đã được lưu vào 'tfidf_weights.csv'")

# 10. Phân tích thêm: So sánh Positive vs Negative
print("\n📊 PHÂN TÍCH THEO SENTIMENT:")
print("-" * 60)

positive_reviews = train_df[train_df['sentiment'] == 1]['review']
negative_reviews = train_df[train_df['sentiment'] == 0]['review']

X_pos = tfidf.transform(positive_reviews.values.astype('U'))
X_neg = tfidf.transform(negative_reviews.values.astype('U'))

tfidf_pos_mean = np.asarray(X_pos.mean(axis=0)).ravel()
tfidf_neg_mean = np.asarray(X_neg.mean(axis=0)).ravel()

# Tìm từ đặc trưng cho mỗi sentiment
pos_specific = pd.DataFrame({
    'word': feature_names,
    'tfidf_positive': tfidf_pos_mean,
    'tfidf_negative': tfidf_neg_mean,
    'difference': tfidf_pos_mean - tfidf_neg_mean
}).sort_values('difference', ascending=False)

print("\n🟢 TOP 10 TỪ ĐẶC TRƯNG CHO POSITIVE:")
for idx, row in pos_specific.head(10).iterrows():
    print(f"{row['word']:<20} Pos: {row['tfidf_positive']:.5f}  Neg: {row['tfidf_negative']:.5f}  Diff: {row['difference']:.5f}")

print("\n🔴 TOP 10 TỪ ĐẶC TRƯNG CHO NEGATIVE:")
for idx, row in pos_specific.tail(10).iterrows():
    print(f"{row['word']:<20} Pos: {row['tfidf_positive']:.5f}  Neg: {row['tfidf_negative']:.5f}  Diff: {row['difference']:.5f}")

# 11. Vẽ biểu đồ so sánh Positive vs Negative
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Top Positive-specific words
top_pos = pos_specific.head(15)
ax1.barh(range(len(top_pos)), top_pos['difference'], color='green', alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top_pos)))
ax1.set_yticklabels(top_pos['word'])
ax1.invert_yaxis()
ax1.set_xlabel('TF-IDF Difference (Positive - Negative)')
ax1.set_title('Top 15 Từ Đặc Trưng Cho POSITIVE', fontweight='bold', fontsize=12)
ax1.grid(axis='x', alpha=0.3)

# Top Negative-specific words
top_neg = pos_specific.tail(15).sort_values('difference')
ax2.barh(range(len(top_neg)), abs(top_neg['difference']), color='red', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(top_neg)))
ax2.set_yticklabels(top_neg['word'])
ax2.invert_yaxis()
ax2.set_xlabel('TF-IDF Difference (Negative - Positive)')
ax2.set_title('Top 15 Từ Đặc Trưng Cho NEGATIVE', fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('tfidf_sentiment_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Biểu đồ so sánh sentiment đã được lưu vào 'tfidf_sentiment_comparison.png'")

print("\n🎉 Hoàn thành phân tích TF-IDF!")
print("\n📁 Các file đã tạo:")
print("  • tfidf_analysis.png - 4 biểu đồ tổng quan")
print("  • tfidf_top30.png - Top 30 từ có TF-IDF cao nhất")
print("  • tfidf_sentiment_comparison.png - So sánh Positive vs Negative")
print("  • tfidf_weights.csv - Dữ liệu đầy đủ")