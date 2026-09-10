# Predictive Mathematics for Supply Chain Forecasting

## Overview
This repository contains the architecture and implementation for a multi-horizon retail forecasting system. The project shifts the traditional business-domain approach of supply chain management toward a rigorous predictive mathematics framework, optimizing for forecasting accuracy and computational efficiency at scale. Model training and evaluation are conducted using the Walmart M5 retail dataset.

## System Architecture
The forecasting engine utilizes a **Segmented Routing Architecture** to dynamically handle different inventory profiles:

* **High-Velocity Inventory:** Fast-moving items are routed through deep learning models designed to capture complex temporal dynamics and non-linear covariates.
* **Sparse Inventory:** Slow-moving items with intermittent demand patterns are routed to robust statistical baselines, preventing overfitting and reducing unnecessary compute overhead.

## Models & Methodologies
* **Temporal Fusion Transformers (TFT):** Deployed for interpretable, multi-horizon predictions and analyzing complex multi-variable relationships.
* **N-BEATS:** Utilized for pure deep learning-based univariate time series forecasting.
* **LightGBM:** Applied for high-performance gradient boosting on engineered tabular time-series features.
* **Croston's Method:** Serves as the primary statistical baseline specifically tailored for intermittent demand.

## Computational Optimizations
Managing the scale of the dataset requires significant optimization to overcome standard hardware bottlenecks:

* **Data Pipeline Engineering:** Streamlined data loading and preprocessing to accelerate batching for large-scale datasets.
* **Precision Tuning:** Implementation of optimized precision settings to manage memory constraints and reduce hardware footprint without sacrificing accuracy.
* **Compute Allocation:** The segmented routing approach naturally balances the computational load by reserving expensive neural network operations for high-impact inventory.

## Setup & Installation
