import gradio as gr

def greet(name):
    return f"Hello {name}! Welcome to the AHO Demo."

def show_info():
    return """# AHO Demo - Adaptive Hybrid Orchestration

This is a placeholder app. The full AHO simulation will be loaded soon.

## What is AHO?
Adaptive Hybrid Orchestration (AHO) is a dynamic workload allocation mechanism for human-AI hybrid systems.

**Features:**
- Hierarchical Uncertainty Quantifier (HUQ)
- Dynamic Threshold Adaptation  
- Fatigue Modeling
- Interactive Controls

The full app is being deployed...
"""

demo = gr.Blocks(title="AHO Demo")
with demo:
    gr.Markdown("# 🤖 AHO Demo - Adaptive Hybrid Orchestration")
    gr.Markdown("Welcome to the Adaptive Hybrid Orchestration simulation!")
    
    with gr.Row():
        name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
        greet_btn = gr.Button("Greet Me")
    
    output = gr.Textbox(label="Greeting", interactive=False)
    info_btn = gr.Button("Show App Info")
    info_output = gr.Markdown()
    
    greet_btn.click(fn=greet, inputs=name_input, outputs=output)
    info_btn.click(fn=show_info, inputs=[], outputs=info_output)

demo.launch()