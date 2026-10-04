import gradio as gr
import logging

from agent import Agent
from agent.embeddings import EMBEDDINGS, cosine_similarity, embed_documents
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
        yield "", history, agent
        return
    if agent is None:
        agent = _make_agent(model_label, temperature, max_tokens)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield "", history, agent
    for partial in agent.stream(message):
        history = history[:-1] + [{"role": "assistant", "content": partial}]
        yield "", history, agent


def on_settings_change(model_label, temperature, max_tokens):
    return _make_agent(model_label, temperature, max_tokens), []


def clear(model_label, temperature, max_tokens):
    return "", [], _make_agent(model_label, temperature, max_tokens)


EMBED_CHOICES = {
    "百炼 / text-embedding-v3": "bailian",
    "OpenAI / text-embedding-3-small": "openai",
    "Ollama / qwen2.5:3b": "ollama",
}


def compare_embeddings(left, right, embed_label):
    left = (left or "").strip()
    right = (right or "").strip()
    if not left or not right:
        return "请填写两段文本"
    provider = EMBED_CHOICES[embed_label]
    logging.getLogger("agent").info("嵌入输入: %s | %s", left, right)
    vectors = embed_documents([left, right], provider=provider)
    score = cosine_similarity(vectors[0], vectors[1])
    logging.getLogger("agent").info("余弦相似度: %.3f", score)
    model = EMBEDDINGS[provider]["model"]
    return f"模型: {model}\n维度: {len(vectors[0])}\n余弦相似度: {score:.3f}"


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

    gr.Markdown("## 文本嵌入")
    embed_dd = gr.Dropdown(
        choices=list(EMBED_CHOICES.keys()),
        value="百炼 / text-embedding-v3",
        label="嵌入模型",
    )
    with gr.Row():
        embed_left = gr.Textbox(label="文本 A", value="我爱你")
        embed_right = gr.Textbox(label="文本 B", value="I love you")
    embed_btn = gr.Button("比较相似度")
    embed_out = gr.Textbox(label="结果", lines=3)
    embed_btn.click(
        compare_embeddings,
        [embed_left, embed_right, embed_dd],
        embed_out,
    )

if __name__ == "__main__":
    demo.launch()
