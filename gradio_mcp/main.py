# This is a sample Python script.

import io
import contextlib

def run_user_script(script_code: str) -> str:
    from tnreason import engine, representation, reasoning, application

    buffer = io.StringIO()

    # Prepare globals dict with imported modules so user script can access them
    user_globals = {
        "engine": engine,
        "representation": representation,
        "reasoning": reasoning,
        "application": application,
        # Optionally also provide 'tnreason' itself if needed:
        "tnreason": __import__("tnreason"),
    }

    try:
        with contextlib.redirect_stdout(buffer):
            exec(script_code, user_globals)
    except Exception as e:
        return f"❌ Error during execution: {e}"

    return buffer.getvalue()



import gradio as gr

def run_script(code):
    return run_user_script(code)

gr.Interface(
    fn=run_script,
    inputs=gr.Textbox(lines=20, label="tnreason script"),
    outputs="text",
    title="MCP for tnreason",
    description="Run your tnreason scripts here. "
).launch()
