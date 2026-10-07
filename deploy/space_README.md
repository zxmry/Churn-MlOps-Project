---
title: Churn MLOps API
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Customer churn model with MLflow registry, CI and drift monitoring
---

# Churn MLOps API

Live API for a customer-churn model: for an online-retail customer, predict the probability of
no purchase in the next 90 days.

- **Try it:** open `/docs` and use **POST /predict → Try it out**. The example body is prefilled.
- **Source, results and write-up:** https://github.com/zxmry/Churn-MlOps-Project

This Space is deployed automatically by GitHub Actions after CI passes on `main`.
The model is the current `@champion` exported from the MLflow registry (see `model/meta.json`).
Prediction logs are kept in the container only and reset when the Space restarts.
