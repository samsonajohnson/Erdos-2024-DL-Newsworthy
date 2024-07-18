# Headlines and Market Trends: Exploring Causality between News Sentiment and Stock Movement Prediction

## Motivation
Financial markets are often affected by sentiment conveyed in news headlines. Understanding the causal relationship between news sentiment and stock price movements can provide deeper insight into market dynamics.

## Goal
Investigate the causal effects between news sentiments and stock price movements. This includes predicting stock movement trends based on news sentiment analysis and understanding how stock movement changes based on future news sentiment. This project aims to study these effects to improve stock movement predictions and optimize portfolio performance.

## Proposed Ideas for the Project
1. **Skip the data gathering step and proceed to data pre-processing**
    - Collected 5 years' worth of financial news and stock data.
    - Improve on the sentiment analysis tool used (FinVader package) by building a modified version and exploring transformer models (e.g., Hugging Face’s ROBERTA pre-trained model) to get more accurate sentiment scores.

2. **Explore Bi-directional Models and CNN**
    - Look at BDLSTM and BD transformer models.
    - Look at converting time series data into an image using Gramian Angular Fields (GAF) method and use as inputs to CNN.

3. **Refine Baseline Model (used ARIMA in the last project)**

4. **Refine simulation of trading strategy that is used to calculate average percentage of portfolio growth – did our models make profit?**

