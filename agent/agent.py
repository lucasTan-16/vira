"""
Agent Loop（asyncio 手写，不引 LangChain）。

作用：编排 LLM 与工具——
  用户提问 → 模型输出
    若含 <tool_call>：解析出工具名和参数 → 执行 → 结果以 tool 角色回传 → 再判断
    若不含：作为最终自然语言回复返回。
阻塞的 rkllm_run 用 asyncio.to_thread 放到线程，不卡住事件循环。
"""
import json
import re
import asyncio

from rkllm_runtime import RKLLMRuntime
from tools import build_registry

# 匹配 <tool_call>{"name":...,"arguments":...}</tool_call>
# re.DOTALL 让 . 能匹配换行，.*? 非贪婪匹配到第一个闭合标签
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# 系统提示：约束模型身份和行为
SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are an industrial "
    "safety inspection assistant running on an edge device. When you "
    "need to inspect the site, query state, or control a device, call "
    "the provided tool. Reply in the user's language."
)

MAX_ITER = 5   # 单次提问最多工具调用轮数，防止无限循环


def parse_tool_calls(text):
    """
    从模型输出中解析工具调用。
    输入：text，模型输出的完整文本
    输出：[(name, arguments dict), ...]；没有工具调用则返回 []
    """
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):          # 个别模型把参数也写成字符串
                args = json.loads(args)
            if name:
                calls.append((name, args))
        except json.JSONDecodeError:
            continue                           # 解析失败的片段跳过
    return calls


class Agent:
    """端侧巡检 Agent：持有 LLM 运行时和工具注册表，提供多轮对话。"""

    def __init__(self, lib_path, model_path, max_context=4096):
        """
        输入：lib_path（librkllmrt.so）、model_path（.rkllm）、max_context
        只创建对象，不加载模型（启动在 start()）。
        """
        self.llm = RKLLMRuntime(lib_path, model_path, max_context)
        self.registry = build_registry()

    def start(self):
        """初始化模型并注册工具。在对话前调用一次。"""
        self.llm.init()
        self.llm.set_function_tools(SYSTEM_PROMPT, self.registry.tools_json())

    async def chat(self, user_text):
        """
        处理一次用户提问，可能包含多轮工具调用。
        输入：user_text，用户问题
        输出：最终自然语言回复（字符串）
        """
        # 第一轮：用户输入；to_thread 把同步阻塞调用放到线程
        raw = await asyncio.to_thread(self.llm.run, "user", user_text)

        for _ in range(MAX_ITER):
            calls = parse_tool_calls(raw)
            if not calls:
                # 没有工具调用，模型输出即最终回复
                return raw.strip()

            # 执行本轮所有工具调用，结果按顺序收集
            results = []
            for name, args in calls:
                res = await asyncio.to_thread(
                    self.registry.execute, name, args)
                results.append(res)

            # 结果以 tool 角色回传，模型再决定继续调工具还是给最终回复
            raw = await asyncio.to_thread(
                self.llm.run, "tool",
                json.dumps(results, ensure_ascii=False))

        # 达到最大轮数仍未结束，返回最后输出（兜底）
        return raw.strip()

    def close(self):
        """释放模型和 NPU 资源。退出前调用。"""
        self.llm.destroy()

