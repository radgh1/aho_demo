import os
from huggingface_hub import HfApi

readme_content = """---
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
"""

with open('temp_readme.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

api = HfApi(token=os.getenv('HF_TOKEN'))
api.upload_file(path_or_fileobj='temp_readme.md', path_in_repo='README.md', repo_id='raddev1/aho_demo', repo_type='space')
print('README.md updated successfully')