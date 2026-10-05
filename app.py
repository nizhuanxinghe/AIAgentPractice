from pathlib import Path

import gradio as gr

from agent import Agent
from agent.documents import get_document_store
from agent.llm import MODEL_CHOICES

DEFAULT_MODEL = "Ollama / qwen2.5:3b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

CSS = """
.gradio-container {
    max-width: min(1400px, 96vw) !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 16px 24px 24px !important;
    background: #eef1f5 !important;
}
#app-shell {
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    gap: 20px !important;
}
#side-panel {
    min-width: 280px !important;
    max-width: 320px !important;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 16px !important;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
}
#side-panel .panel-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 12px;
}
#side-panel {
    flex: 0 0 300px !important;
    position: sticky !important;
    top: 16px !important;
}
#main-panel {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    max-width: 100% !important;
}
#chat-panel .bubble-wrap {
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
}
#chat-panel,
#chat-panel .message,
#chat-panel .message-wrap {
    max-width: 100% !important;
    min-width: 0 !important;
}
#chat-panel .message {
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}
#chat-panel pre {
    max-width: 100% !important;
    overflow-x: auto !important;
    white-space: pre-wrap !important;
}
#page-header {
    text-align: left;
    padding: 4px 0 12px;
}
#page-header h1 {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0;
}
#page-header p {
    font-size: 0.85rem;
    color: #9ca3af;
    margin: 6px 0 0;
}
#chat-panel {
    height: max(320px, calc(100vh - 380px)) !important;
    max-height: max(320px, calc(100vh - 380px)) !important;
    min-height: 0 !important;
    overflow: hidden !important;
    border: none !important;
    box-shadow: none !important;
    background: #fff !important;
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
}
#chip-row {
    flex-wrap: wrap;
    gap: 10px;
    margin: 12px 0;
}
#input-dock {
    width: 100% !important;
    background: #fff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 20px !important;
    padding: 10px 12px 8px !important;
    overflow: visible !important;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05) !important;
}
#input-dock:focus-within {
    border-color: #eef1f5 !important;
    box-shadow: 0 0 0 1px #eef1f5 !important;
}
#input-dock-wrap {
    gap: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#input-dock textarea {
    min-height: 56px !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 0.95rem !important;
    padding: 6px 8px 4px !important;
    resize: none !important;
}
#input-dock textarea:focus,
#input-dock textarea:focus-visible {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
}
.chip-btn button {
    border-radius: 999px !important;
    background: #eef0f3 !important;
    border: none !important;
    color: #374151 !important;
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
}
.chip-btn button:hover {
    background: #e5e7eb !important;
}
#send-btn {
    margin: 0 !important;
    width: auto !important;
    min-height: 0 !important;
}
#send-btn button {
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    border-radius: 999px !important;
    background: transparent !important;
    border: 1.5px solid #d1d5db !important;
    color: #374151 !important;
    font-size: 1.5rem !important;
    line-height: 1 !important;
    padding: 0 !important;
    box-shadow: none !important;
}
#send-btn button:hover {
    background: transparent !important;
}
#input-dock,
#input-dock > *,
#input-dock-wrap,
#input-dock-wrap > *,
#input-dock-toolbar,
#input-dock-toolbar > *,
#input-dock-toolbar .block,
#input-dock-toolbar .form,
#input-dock-toolbar .column,
#attach-anchor,
#attach-anchor > * {
    --block-background-fill: transparent;
    --background-fill-primary: transparent;
    --background-fill-secondary: transparent;
    --border-color-primary: transparent;
    box-shadow: none !important;
}
#input-dock-toolbar,
#input-dock-toolbar > *,
#input-dock-toolbar .block,
#input-dock-toolbar .form,
#attach-anchor > .block {
    background: transparent !important;
    border: none !important;
}
#input-dock-toolbar {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    width: 100% !important;
    padding: 2px 2px 4px !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    box-sizing: border-box !important;
}
#input-dock-toolbar > .column:has(#send-btn),
#input-dock-toolbar > .form > .column:has(#send-btn),
#input-dock-toolbar .block:has(#send-btn) {
    margin-left: auto !important;
    flex: 0 0 auto !important;
    width: auto !important;
}
.dock-toolbar-spacer {
    flex: 1 1 auto;
    min-width: 8px;
}
#attach-anchor {
    position: relative !important;
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    gap: 0 !important;
}
#attach-plus-btn button {
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    border-radius: 999px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #374151 !important;
    font-size: 1.65rem !important;
    font-weight: 400 !important;
    line-height: 1 !important;
    padding: 0 !important;
}
#attach-plus-btn button:hover {
    background: transparent !important;
}
#attach-menu {
    position: absolute !important;
    left: 0 !important;
    bottom: calc(100% + 8px) !important;
    z-index: 20 !important;
    min-width: 140px !important;
    padding: 6px !important;
    background: #fff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    gap: 4px !important;
}
.attach-menu-item button {
    width: 100% !important;
    justify-content: flex-start !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #111827 !important;
    font-size: 0.88rem !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
}
.attach-menu-item button:hover {
    background: #eef1f5 !important;
}
#dock-model {
    flex: 0 0 auto !important;
    width: 210px !important;
    min-width: 0 !important;
    max-width: 210px !important;
    margin: 0 !important;
}
#dock-model ul.options,
#dock-model .options {
    background: #fff !important;
    opacity: 1 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    z-index: 30 !important;
}
#dock-model .options li:hover,
#dock-model .options li.active {
    background: #eef1f5 !important;
}
#dock-model label,
#dock-model .label-wrap {
    display: none !important;
}
#dock-model,
#dock-model .wrap,
#dock-model .wrap-inner,
#dock-model .secondary-wrap,
#dock-model select,
#dock-model input {
    min-height: 32px !important;
    font-size: 0.85rem !important;
    border-radius: 999px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#doc-file-input {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
}
#new-chat-btn button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 1.1rem;
}
"""

SUGGESTIONS = [
    "现在几点？",
    "查询北京天气",
    "武汉的城市信息和今天天气",
]


def _make_agent(
    model_label: str,
    temperature: float,
    max_tokens: int,
    doc_enabled: bool = False,
) -> Agent:
    provider, model = MODEL_CHOICES[model_label]
    agent = Agent(
        provider=provider,
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    if doc_enabled:
        agent.enable_documents()
    return agent


def _reply(history, message, text):
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": text},
    ]


def respond(
    message,
    history,
    agent,
    model_label,
    temperature,
    max_tokens,
    doc_file,
    doc_enabled,
):
    message = (message or "").strip()
    history = list(history or [])
    if doc_file and not message:
        history = history + [
            {"role": "assistant", "content": "请在输入框写明要求，例如学习或分析"},
        ]
        yield "", history, agent, doc_file, doc_enabled
        return
    if not message:
        yield "", history, agent, doc_file, doc_enabled
        return
    if agent is None:
        agent = _make_agent(model_label, temperature, max_tokens, doc_enabled)
    if doc_file:
        path = doc_file if isinstance(doc_file, str) else str(doc_file)
        if Path(path).suffix.lower() != ".txt":
            yield "", _reply(history, message, "只支持 txt 文档"), agent, None, doc_enabled
            return
        try:
            get_document_store().ingest(path)
            agent.enable_documents()
            doc_enabled = True
        except Exception as exc:
            yield "", _reply(history, message, f"错误：{exc}"), agent, None, doc_enabled
            return
    history = _reply(history, message, "")
    yield "", history, agent, None, doc_enabled
    for partial in agent.stream(message):
        history = history[:-1] + [{"role": "assistant", "content": partial}]
        yield "", history, agent, None, doc_enabled


def on_settings_change(model_label, temperature, max_tokens, doc_enabled):
    return _make_agent(model_label, temperature, max_tokens, doc_enabled), [], doc_enabled


def toggle_attach_menu(open_):
    open_ = not open_
    return open_, gr.update(visible=open_)


def close_attach_menu(_path):
    return gr.update(visible=False), False


def toggle_send_btn(text):
    return gr.update(visible=bool((text or "").strip()))


def dismiss_attach_menu():
    return gr.update(visible=False), False


def clear(model_label, temperature, max_tokens, doc_enabled):
    return (
        "",
        [],
        _make_agent(model_label, temperature, max_tokens, False),
        False,
        None,
        gr.update(visible=False),
        False,
    )


with gr.Blocks(title="AI Agent") as demo:
    agent_state = gr.State(None)
    doc_enabled = gr.State(False)

    with gr.Row(elem_id="app-shell"):
        with gr.Column(elem_id="side-panel", scale=0, min_width=300):
            gr.HTML("<div class='panel-title'>设置</div>")
            temperature = gr.Slider(
                0,
                2,
                value=DEFAULT_TEMPERATURE,
                step=0.1,
                label="temperature",
            )
            max_tokens = gr.Slider(
                256,
                4096,
                value=DEFAULT_MAX_TOKENS,
                step=128,
                label="max_tokens",
            )
            new_chat_btn = gr.Button("新对话", elem_id="new-chat-btn")

        with gr.Column(elem_id="main-panel", scale=1):
            with gr.Row():
                with gr.Column(elem_id="page-header"):
                    gr.Markdown("# AI Agent")
                    gr.Markdown("AI 生成可能有误，请核实")

            chatbot = gr.Chatbot(
                elem_id="chat-panel",
                height="max(320px, calc(100vh - 380px))",
                placeholder="输入问题开始对话",
                show_label=False,
            )

            chip_buttons = []
            with gr.Row(elem_id="chip-row"):
                for tip in SUGGESTIONS:
                    chip = gr.Button(f"{tip} →", elem_classes="chip-btn", size="sm")
                    chip_buttons.append(chip)

            with gr.Group(elem_id="input-dock"):
                with gr.Column(elem_id="input-dock-wrap"):
                    msg = gr.Textbox(
                        placeholder="发消息…",
                        show_label=False,
                        lines=2,
                        max_lines=8,
                        container=False,
                    )
                    with gr.Row(elem_id="input-dock-toolbar"):
                        with gr.Column(elem_id="attach-anchor", scale=0, min_width=48):
                            attach_plus_btn = gr.Button(
                                "+",
                                elem_id="attach-plus-btn",
                                scale=0,
                                min_width=48,
                            )
                            with gr.Column(elem_id="attach-menu", visible=False) as attach_menu:
                                file_pick_btn = gr.Button(
                                    "文件选择",
                                    elem_classes="attach-menu-item",
                                    size="sm",
                                )
                        model_dd = gr.Dropdown(
                            choices=list(MODEL_CHOICES.keys()),
                            value=DEFAULT_MODEL,
                            show_label=False,
                            container=False,
                            elem_id="dock-model",
                            scale=0,
                        )
                        gr.HTML("<div class='dock-toolbar-spacer'></div>")
                        send_btn = gr.Button(
                            "↑",
                            elem_id="send-btn",
                            scale=0,
                            min_width=48,
                            visible=False,
                        )
                doc_file = gr.File(
                    visible=False,
                    elem_id="doc-file-input",
                    file_types=[".txt"],
                    type="filepath",
                )

    attach_open = gr.State(False)
    attach_dismiss = gr.Button("close", visible=False, elem_id="attach-dismiss")
    settings = [model_dd, temperature, max_tokens]
    outputs = [msg, chatbot, agent_state, doc_file, doc_enabled]

    for ctrl in settings:
        ctrl.change(on_settings_change, [*settings, doc_enabled], [agent_state, chatbot, doc_enabled])
    msg.submit(respond, [msg, chatbot, agent_state, *settings, doc_file, doc_enabled], outputs)
    send_btn.click(respond, [msg, chatbot, agent_state, *settings, doc_file, doc_enabled], outputs)
    new_chat_btn.click(
        clear,
        [*settings, doc_enabled],
        [msg, chatbot, agent_state, doc_enabled, doc_file, attach_menu, attach_open],
    )

    msg.change(toggle_send_btn, msg, send_btn, queue=False, show_progress="hidden")

    attach_plus_js = """
    (e) => {
      if (e && e.stopPropagation) e.stopPropagation();
    }
    """
    attach_plus_btn.click(
        toggle_attach_menu,
        attach_open,
        [attach_open, attach_menu],
        js=attach_plus_js,
    )
    attach_dismiss.click(dismiss_attach_menu, None, [attach_menu, attach_open])
    file_pick_js = """
    () => {
      const root = document.querySelector("#doc-file-input");
      const input = root && root.querySelector('input[type="file"]');
      if (input) input.click();
    }
    """
    file_pick_btn.click(None, None, None, js=file_pick_js)
    doc_file.change(close_attach_menu, doc_file, [attach_menu, attach_open])

    for chip, tip in zip(chip_buttons, SUGGESTIONS):
        chip.click(lambda t=tip: t, None, msg)

    dock_js = """
    () => {
      if (!window.__dockAttachOutside) {
        window.__dockAttachOutside = true;
        document.addEventListener("click", (e) => {
          const menu = document.querySelector("#attach-menu");
          const plus = document.querySelector("#attach-plus-btn");
          const dismiss = document.querySelector("#attach-dismiss button");
          if (!menu || !dismiss) return;
          if (menu.offsetParent === null && menu.style.display === "none") return;
          const hidden = menu.closest('[style*="display: none"]');
          if (hidden && hidden !== menu) return;
          if (plus && plus.contains(e.target)) return;
          if (menu.contains(e.target)) return;
          if (menu.offsetHeight > 0 && window.getComputedStyle(menu).visibility !== "hidden") {
            dismiss.click();
          }
        });
      }
    }
    """
    demo.load(js=dock_js)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
