from agent import Agent


def main() -> None:
    agent = Agent()
    questions = [
        "你好，请用一句话介绍你自己",
        "现在几点了？",
    ]
    for q in questions:
        print(f"\n你: {q}")
        print(f"Agent: {agent.run(q)}")


if __name__ == "__main__":
    main()
