
# 🏨 Matching Booking Users to Their Reviews — Kaggle Solution

This repository contains the solution developed by **Yarin Yerushalmi Levi** and **Idan Yankelev** for the **"Matching Booking Users to Their Reviews"** Kaggle competition.

## 📌 Problem Overview

The challenge involves linking user profiles to their corresponding hotel reviews, even when the test and validation sets contain users and accommodations not present in the training set. This requires a model that generalizes well using the available textual and numerical features.

## 🧠 Approach Summary

Our solution is centered around **contrastive learning**, where we aim to:
- Maximize similarity between correct user-review pairs.
- Minimize similarity for mismatched pairs.

We use **sentence-level embeddings** and **contrastive cross-entropy loss** to learn meaningful representations of users and reviews.

## 📂 Dataset

The dataset includes:
- Over 1 million reviews.
- User profile information.
- Accommodation features.
- Review texts with ratings and feedback.

### Preprocessing Steps:
1. **Handling Missing Values**: Replaced `NaN`s with `"unknown"`.
2. **Date Formatting**: Converted month numbers to names.
3. **Context Generation**: Constructed input strings:
   - **User context example**:
     ```
     "We visited Australia as a family from Zuc for 2 days and 1 night during December and stayed at an accommodation with 3.5-star rating scoring 7.9/10 near beach and near city center."
     ```
   - **Review context example**:
     ```
     "Review titled: Lovely relaxing hotel, great value for money! Positive feedback: The food was lovely and the pool was great. [...] The review gave a score of 10."
     ```

## 🧪 Model Architecture & Training

- **Embedding Model**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Loss Function**: Contrastive cross-entropy loss.
- **Sampling Strategy**: Sampled batches per accommodation to maintain semantic coherence.
- **Batch Sizes Evaluated**: 16–128.
- **Training Strategy**:
  - Sentence-transformer architecture for embeddings.
  - Fine-tuned on sampled batches with user-review pairs.

## 📈 Evaluation

- **Primary Metric**: MRR@10 (Mean Reciprocal Rank at cutoff 10).
- **Model Selection**: Best model chosen based on validation subset performance.

## ✅ Key Insights

- **Representation Matters**: Adjusting the formatting and concatenation of input contexts greatly influenced performance.
- **Loss Function Choice**: Switching to contrastive cross-entropy improved results.
- **Sampling Technique**: Grouping by accommodation allowed better contextual learning.

## 🔮 Future Work

1. **Improve Data Quality**: Advanced cleaning and normalization.
2. **Explore Stronger Models**: Test larger or domain-specific transformers.
3. **Introduce Personalization**: Account for user behavior patterns and preferences.
