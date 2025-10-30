---
title: AHO Demo - Adaptive Hybrid Orchestration
emoji: "🤖"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: Interactive AHO simulation for human-AI decision making
---

# AHO Demo: Adaptive Hybrid Orchestration

This interactive demo showcases **Plan A: Adaptive Hybrid Orchestration (AHO)** - a dynamic workload allocation mechanism for human-AI hybrid systems.

## Features

- **Hierarchical Uncertainty Quantifier (HUQ)**: Computes confidence scores for both AI and human experts
- **Dynamic Threshold Adaptation**: Uses contextual bandit with epsilon-greedy exploration
- **Fatigue Modeling**: Accounts for human expert performance degradation
- **Interactive Controls**: Experiment with different parameters and see real-time results

## How It Works

The system routes tasks between AI and human experts based on uncertainty thresholds that adapt over time. Use the controls to explore how different settings affect the balance between accuracy and efficiency.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


*Last updated: October 30, 2025 - Force rebuild*