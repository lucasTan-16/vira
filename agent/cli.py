"""
命令行入口：板端对话闭环。

作用：加载 LLM 和工具，循环读取用户问题，
     调用 Agent 处理并打印回复，退出时释放 NPU。
运行：python3 cli.py
"""
import os
import asyncio
import argparse

from agent import Agent

HERE = os.path.dirname(os.path.abspath(__file__))   # agent/
ROOT = os.path.dirname(HERE)                       # 工程根

# 默认运行时库和模型路径
DEFAULT_LIB = os.path.join(ROOT, "llm", "librkllmrt.so")
DEFAULT_MODEL = os.path.join(
    ROOT, "llm", "Qwen2.5-0.5B-Instruct_W8A8_RK3588.rkllm")


async def repl(agent):
    """
    交互式对话循环。
    输入：agent（已启动的 Agent）
    输出：逐轮打印模型回复；exit/quit 或 Ctrl+C 结束
    """
    print("VIRA agent ready. 输入问题（exit 退出）。")
    loop = asyncio.get_event_loop()
    while True:
        try:
            # input 阻塞，用 run_in_executor 放到线程，不卡事件循环
            user = await loop.run_in_executor(None, input, "user> ")
        except (EOFError, KeyboardInterrupt):
            break
        user = user.strip()
        if not user:
            continue
        if user in ("exit", "quit"):
            break
        reply = await agent.chat(user)
        print(f"\nrobot> {reply}\n")


def main():
    """解析参数、创建并启动 Agent，运行对话，最后释放。"""
    parser = argparse.ArgumentParser(description="VIRA edge agent")
    parser.add_argument("--lib", default=DEFAULT_LIB,
                        help="librkllmrt.so 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=".rkllm 模型路径")
    args = parser.parse_args()

    agent = Agent(args.lib, args.model)
    agent.start()
    try:
        asyncio.run(repl(agent))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
