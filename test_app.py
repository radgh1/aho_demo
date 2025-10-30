import gradio as gr

def greet(name):
    return f"Hello {name}! This is a test."

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()