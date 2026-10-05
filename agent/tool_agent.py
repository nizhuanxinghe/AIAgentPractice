from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from tools import TOOLS

REACT_TEMPLATE = """你是 AI Agent，用中文回答。需要时可多次调用工具，直到信息足够再给出 Final Answer。
{persona}
可用工具：
{tools}

格式：
Question: 用户问题
Thought: 思考
Action: 工具名，必须是 [{tool_names}] 之一
Action Input: 只写城市名或空字符串，不要写 city= 或 JSON
Observation: 工具返回
... 可重复 Thought/Action/Action Input/Observation ...
Thought: 我已掌握足够信息
Final Answer: 给用户的最终回答

Question: {input}
Thought:{agent_scratchpad}"""


def build_executor(provider: str, model: str | None, temperature: float) -> AgentExecutor:
    from .lc_llm import get_chat_model

    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    llm = get_chat_model(provider, model, temperature)
    agent = create_react_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=5,
        handle_parsing_errors=True,
        verbose=False,
    )
