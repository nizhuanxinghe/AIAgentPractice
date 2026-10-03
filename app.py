import gradio as gr

from agent import Agent
from agent.llm import MODEL_CHOICES

DEFAULT_MODEL = "百炼 / qwen-plus"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024


def _make_agent(model_label: str, temperature: float, max_tokens: int) -> Agent:
    provider, model = MODEL_CHOICES[model_label]
    return Agent(
        provider=provider,
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )


def respond(message, history, agent, model_label, temperature, max_tokens):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return "", history, agent
    if agent is None:
        agent = _make_agent(model_label, temperature, max_tokens)
    reply = agent.run(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history, agent


def on_settings_change(model_label, temperature, max_tokens):
    return _make_agent(model_label, temperature, max_tokens), []


def clear(model_label, temperature, max_tokens):
    return "", [], _make_agent(model_label, temperature, max_tokens)


with gr.Blocks(title="AI Agent") as demo:
    gr.Markdown("## AI Agent")
    agent_state = gr.State(None)
    model_dd = gr.Dropdown(
        choices=list(MODEL_CHOICES.keys()),
        value=DEFAULT_MODEL,
        label="模型",
    )
    with gr.Row():
        temperature = gr.Slider(0, 2, value=DEFAULT_TEMPERATURE, step=0.1, label="temperature")
        max_tokens = gr.Slider(256, 4096, value=DEFAULT_MAX_TOKENS, step=128, label="max_tokens")
    chatbot = gr.Chatbot(height=480, placeholder="输入问题开始对话")
    with gr.Row():
        msg = gr.Textbox(placeholder="输入后回车或点发送", show_label=False, scale=8)
        send_btn = gr.Button("发送", scale=1, variant="primary")
    clear_btn = gr.Button("清空对话")

    settings = [model_dd, temperature, max_tokens]
    for ctrl in settings:
        ctrl.change(on_settings_change, settings, [agent_state, chatbot])
    msg.submit(
        respond,
        [msg, chatbot, agent_state, *settings],
        [msg, chatbot, agent_state],
    )
    send_btn.click(
        respond,
        [msg, chatbot, agent_state, *settings],
        [msg, chatbot, agent_state],
    )
    clear_btn.click(clear, settings, [msg, chatbot, agent_state])

if __name__ == "__main__":
    demo.launch()
