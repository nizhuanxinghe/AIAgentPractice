import gradio as gr

from agent import Agent


def respond(message, history, agent):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return "", history, agent
    if agent is None:
        agent = Agent()
    reply = agent.run(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history, agent


def clear():
    return "", [], None


with gr.Blocks(title="AI Agent") as demo:
    gr.Markdown("## AI Agent")
    agent_state = gr.State(None)
    chatbot = gr.Chatbot(height=480, placeholder="输入问题开始对话")
    with gr.Row():
        msg = gr.Textbox(placeholder="输入后回车或点发送", show_label=False, scale=8)
        send_btn = gr.Button("发送", scale=1, variant="primary")
    clear_btn = gr.Button("清空对话")
    msg.submit(respond, [msg, chatbot, agent_state], [msg, chatbot, agent_state])
    send_btn.click(respond, [msg, chatbot, agent_state], [msg, chatbot, agent_state])
    clear_btn.click(clear, outputs=[msg, chatbot, agent_state])

if __name__ == "__main__":
    demo.launch()
