"""
ctypes 封装 librkllmrt.so（Rockchip RKLLM 运行时 C API）。

作用：让 Python Agent 能在板端加载并运行 .rkllm 模型，
     把 C 的结构体、枚举、回调逐一映射成 Python 可调用对象。
本文件只做"Python 怎么调 C"，不包含 Agent 决策逻辑。
"""
import ctypes
from ctypes import (c_int, c_int8, c_int32, c_uint8, c_uint32, c_float,
                    c_bool, c_char_p, c_size_t, c_void_p, byref, Structure,
                    Union, POINTER, CFUNCTYPE)

# ===== 枚举常量（对应 rkllm.h 中的 enum）=====
# 回调状态 LLMCallState：模型每生成一段/结束/出错时回调的状态
RKLLM_RUN_NORMAL = 0    # 正常：本次回调带一段新生成的文本
RKLLM_RUN_FINISH = 2    # 结束：本轮推理完成，此时 perf 字段有效
RKLLM_RUN_ERROR = 3     # 出错

# 输入类型 RKLLMInputType：告诉 rkllm 这次喂的是文本、token 还是 embedding
RKLLM_INPUT_PROMPT = 0  # 文本字符串（最常用）
RKLLM_INPUT_TOKEN = 1   # 已分词的 token id 数组
RKLLM_INPUT_EMBED = 2   # 已算好的 embedding 向量

# 推理模式 RKLLMInferMode
RKLLM_INFER_GENERATE = 0            # 正常生成文本
RKLLM_INFER_GET_LAST_HIDDEN = 1     # 只取最后一层隐藏状态（不生成）


class RKLLMExtendParam(Structure):
    """扩展参数：绑核、批大小等底层调度配置（对应 C 的 RKLLMExtendParam）。"""
    _fields_ = [
        ("base_domain_id", c_int32),    # 预留，固定 0
        ("embed_flash", c_int8),        # 1=大词表 embedding 用 flash 加载（省内存，默认 1）
        ("enabled_cpus_num", c_int8),   # 参与计算的 CPU 核数
        ("enabled_cpus_mask", c_uint32),# 参与 CPU 核的位掩码（RK3588 大核 4-7 为 0xf0）
        ("n_batch", c_uint8),           # 多请求批大小，默认 1
        ("use_cross_attn", c_int8),     # 是否用 cross attention（自定义模型才用）
        ("reserved", c_uint8 * 104),    # 预留字段，必须保留（对齐 C 内存布局）
    ]


class RKLLMParam(Structure):
    """模型初始化参数（输入）：rkllm_init 时传入，决定模型路径和采样默认值。"""
    _fields_ = [
        ("model_path", c_char_p),       # 输入：.rkllm 模型文件路径（字节串）
        ("max_context_len", c_int32),   # 输入：最大上下文长度（由模型决定，这里填 4096）
        ("max_new_tokens", c_int32),    # 输入：单轮最多生成多少 token（必须为正，如 512）
        ("top_k", c_int32),             # 输入：top-k 采样，只在概率最高的 k 个 token 里选（如 5）
        ("n_keep", c_int32),            # 输入：对话中保留的前缀 token 数，默认 0
        ("top_p", c_float),             # 输入：核采样阈值，累计概率到 p 为止（如 0.9）
        ("temperature", c_float),       # 输入：温度，越高越随机（如 0.8；0 表示贪心）
        ("repeat_penalty", c_float),    # 输入：重复惩罚，>1 抑制重复（如 1.1）
        ("frequency_penalty", c_float), # 输入：频率惩罚
        ("presence_penalty", c_float),  # 输入：存在惩罚
        ("mirostat", c_int32),          # 输入：mirostat 采样开关，0 关闭
        ("mirostat_tau", c_float),      # 输入：mirostat 目标熵
        ("mirostat_eta", c_float),      # 输入：mirostat 学习率
        ("skip_special_token", c_bool), # 输入：是否跳过特殊 token（如 <|im_end|>），通常 True
        ("is_async", c_bool),           # 输入：是否异步推理，这里 False（同步）
        ("img_start", c_char_p),        # 输入：多模态图像起始标记（纯文本模型不用）
        ("img_end", c_char_p),          # 输入：多模态图像结束标记
        ("img_content", c_char_p),      # 输入：多模态图像占位标记
        ("extend_param", RKLLMExtendParam),  # 输入：扩展参数（绑核/flash）
    ]


class RKLLMEmbedInput(Structure):
    """embedding 输入：直接喂向量（input_type=EMBED 时用）。"""
    _fields_ = [
        ("embed", POINTER(c_float)),    # 输入：指向 embedding 数组的指针
        ("n_tokens", c_size_t),         # 输入：向量对应的 token 数
    ]


class RKLLMTokenInput(Structure):
    """token 输入：直接喂已分词的 id（input_type=TOKEN 时用）。"""
    _fields_ = [
        ("input_ids", POINTER(c_int32)),# 输入：token id 数组指针
        ("n_tokens", c_size_t),         # 输入：token 个数
    ]


class RKLLMMultiModelInput(Structure):
    """多模态输入：文本 + 图像（本项目用纯文本，暂不使用）。"""
    _fields_ = [
        ("prompt", c_char_p),           # 输入：文本提示
        ("image_embed", POINTER(c_float)),  # 输入：图像 embedding
        ("n_image_tokens", c_size_t),   # 输入：图像 token 数
        ("n_image", c_size_t),          # 输入：图像数量
        ("image_width", c_size_t),      # 输入：图像宽
        ("image_height", c_size_t),     # 输入：图像高
    ]


class RKLLMInputUnion(Union):
    """输入联合体：根据 input_type 只使用其中一个成员（对应 C 的匿名 union）。"""
    _fields_ = [
        ("prompt_input", c_char_p),         # 文本输入（input_type=PROMPT 时用）
        ("embed_input", RKLLMEmbedInput),   # embedding 输入
        ("token_input", RKLLMTokenInput),   # token 输入
        ("multimodal_input", RKLLMMultiModelInput),  # 多模态输入
    ]


class RKLLMInput(Structure):
    """一次推理的输入（输入）：角色 + 内容；rkllm_run 每次调用都要传。"""
    _fields_ = [
        ("role", c_char_p),             # 输入：角色，"user"/"assistant"/"system"/"tool"
        ("enable_thinking", c_bool),    # 输入：是否开启思考模式（Qwen-Instruct 填 False）
        ("input_type", c_int32),        # 输入：内容类型，取上面的 RKLLM_INPUT_* 常量
        ("u", RKLLMInputUnion),         # 输入：具体内容，按 input_type 选 union 成员
    ]


class RKLLMLoraParam(Structure):
    """LoRA 适配器参数（本项目不用 LoRA，保留定义）。"""
    _fields_ = [("lora_adapter_name", c_char_p)]  # 输入：LoRA 适配器名


class RKLLMPromptCacheParam(Structure):
    """prompt cache 参数：保存/复用前缀缓存，加速重复前缀（本项目暂不用）。"""
    _fields_ = [
        ("save_prompt_cache", c_int),   # 输入：1=保存前缀缓存，0=不保存
        ("prompt_cache_path", c_char_p),# 输入：缓存文件路径
    ]


class RKLLMInferParam(Structure):
    """推理控制参数（输入）：模式、是否保留历史等。"""
    _fields_ = [
        ("mode", c_int32),                          # 输入：推理模式，取 RKLLM_INFER_*
        ("lora_params", POINTER(RKLLMLoraParam)),   # 输入：LoRA 参数（不用则 NULL）
        ("prompt_cache_params", POINTER(RKLLMPromptCacheParam)),  # 输入：缓存参数（不用则 NULL）
        ("keep_history", c_int),                    # 输入：1=保留对话历史，0=不保留
    ]


class RKLLMResultLastHiddenLayer(Structure):
    """输出：最后一层隐藏状态（mode=GET_LAST_HIDDEN 时有效）。"""
    _fields_ = [
        ("hidden_states", POINTER(c_float)),  # 输出：隐藏层向量指针
        ("embd_size", c_int),                 # 输出：每个 token 的向量维度
        ("num_tokens", c_int),                # 输出：token 个数
    ]


class RKLLMResultLogits(Structure):
    """输出：logits（原始词表概率，mode=GET_LOGITS 时有效）。"""
    _fields_ = [
        ("logits", POINTER(c_float)),   # 输出：logits 数组指针
        ("vocab_size", c_int),          # 输出：词表大小
        ("num_tokens", c_int),          # 输出：token 个数
    ]


class RKLLMPerfStat(Structure):
    """输出：性能统计，仅在回调状态 FINISH 时有效。"""
    _fields_ = [
        ("prefill_time_ms", c_float),   # 输出：prefill（处理输入）耗时，毫秒
        ("prefill_tokens", c_int),      # 输出：输入 token 数
        ("generate_time_ms", c_float),  # 输出：生成阶段总耗时，毫秒
        ("generate_tokens", c_int),     # 输出：生成 token 数
        ("memory_usage_mb", c_float),   # 输出：峰值内存占用，MB
    ]


class RKLLMResult(Structure):
    """回调结果（输出）：每次回调由 C 端填充，包含文本和统计。"""
    _fields_ = [
        ("text", c_char_p),                        # 输出：本次新增的文本片段（NORMAL 时）
        ("token_id", c_int32),                     # 输出：本次生成的 token id
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),  # 输出：隐藏层
        ("logits", RKLLMResultLogits),             # 输出：logits
        ("perf", RKLLMPerfStat),                   # 输出：性能统计（FINISH 时）
    ]

# 回调函数类型，对应 C 的：
#   int (*LLMResultCallback)(RKLLMResult *result, void *userdata, LLMCallState state)
# CFUNCTYPE 第一个参数是返回类型，后面依次是三个参数类型
LLMResultCallback = CFUNCTYPE(
    c_int, POINTER(RKLLMResult), c_void_p, c_int32
)


class RKLLMRuntime:
    """板端 RKLLM 运行时：加载 .so，初始化模型，提供同步 run 接口。"""

    def __init__(self, lib_path, model_path, max_context=4096):
        """
        作用：只加载库、声明签名，不真正初始化模型（初始化在 init()）。
        输入：
            lib_path:    librkllmrt.so 路径（如 /home/orangepi/vira/llm/librkllmrt.so）
            model_path:  .rkllm 模型路径
            max_context: 最大上下文长度（与转换时一致，4096）
        """
        self.model_path = model_path
        self.max_context = max_context
        # CDLL 加载普通 C 动态库
        self.lib = ctypes.CDLL(lib_path)
        self._declare_signatures()
        self.handle = None          # init 成功后填充模型句柄
        self._chunks = []          # 本轮回调收集的文本片段
        # 必须保持回调对象引用，否则被 Python 回收后 C 端回调会崩溃
        self._cb = LLMResultCallback(self._on_result)

    def _declare_signatures(self):
        """声明每个 C 函数的参数类型和返回类型，ctypes 才能正确转换。"""
        lib = self.lib
        # rkllm_init(LLMHandle *handle, RKLLMParam *param, callback)
        lib.rkllm_init.argtypes = [
            POINTER(c_void_p), POINTER(RKLLMParam), LLMResultCallback
        ]
        lib.rkllm_init.restype = c_int
        # rkllm_run(handle, RKLLMInput *input, RKLLMInferParam *param, userdata)
        lib.rkllm_run.argtypes = [
            c_void_p, POINTER(RKLLMInput), POINTER(RKLLMInferParam), c_void_p
        ]
        lib.rkllm_run.restype = c_int
        # rkllm_destroy(handle)
        lib.rkllm_destroy.argtypes = [c_void_p]
        lib.rkllm_destroy.restype = c_int
        # rkllm_set_function_tools(handle, system_prompt, tools, tool_response_str)
        lib.rkllm_set_function_tools.argtypes = [
            c_void_p, c_char_p, c_char_p, c_char_p
        ]
        lib.rkllm_set_function_tools.restype = c_int
        # rkllm_createDefaultParam()：无参数，按值返回一个 RKLLMParam
        lib.rkllm_createDefaultParam.argtypes = []
        lib.rkllm_createDefaultParam.restype = RKLLMParam

    def _on_result(self, result_ptr, userdata, state):
        """
        C 回调（rkllm 在内部逐段调用）：按状态收集文本。
        输入：result_ptr（结果指针）、userdata（用户数据，未用）、state（状态）
        输出：固定返回 0，表示继续推理
        """
        result = result_ptr.contents
        if state == RKLLM_RUN_NORMAL and result.text:
            # C 的 char* 是 UTF-8 字节，解码为 Python 字符串
            self._chunks.append(result.text.decode("utf-8", errors="replace"))
        elif state == RKLLM_RUN_ERROR:
            self._chunks.append("[rkllm error]")
        # FINISH 时无需处理文本
        return 0

    def init(self, top_k=5, top_p=0.9, temperature=0.8):
        """
        初始化模型：填充 RKLLMParam，调用 rkllm_init。
        输入：采样默认值 top_k / top_p / temperature
        输出：成功返回 0；失败抛异常
        """
        param = RKLLMParam()
        ctypes.memset(byref(param), 0, ctypes.sizeof(param))  # 先清零
        param.model_path = self.model_path.encode("utf-8")
        param.max_context_len = self.max_context
        param.max_new_tokens = 512       # 单轮最多生成 512 token
        param.top_k = top_k
        param.top_p = top_p
        param.temperature = temperature
        param.repeat_penalty = 1.1
        param.skip_special_token = True
        param.is_async = False
        param.extend_param.embed_flash = 1   # 大词表 flash 加载，省内存
        param.extend_param.n_batch = 1       # 批大小，必须 1~100


        self.handle = c_void_p()
        ret = self.lib.rkllm_init(byref(self.handle), byref(param), self._cb)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed, ret={ret}")
        return ret

    def set_function_tools(self, system_prompt, tools_json,
                           tool_response_str="tool_response"):
        """
        注册 function calling 工具。
        输入：system_prompt（系统提示）、tools_json（工具定义 JSON）、
              tool_response_str（工具响应标识，用于区分普通对话）
        输出：成功返回 0；失败抛异常
        """
        ret = self.lib.rkllm_set_function_tools(
            self.handle,
            system_prompt.encode("utf-8"),
            tools_json.encode("utf-8"),
            tool_response_str.encode("utf-8"),
        )
        if ret != 0:
            raise RuntimeError(f"set_function_tools failed, ret={ret}")
        return ret

    def run(self, role, text, keep_history=1, input_type=RKLLM_INPUT_PROMPT):
        """
        跑一轮推理（同步，阻塞到 FINISH）。
        输入：
            role:         角色，"user"/"assistant"/"system"/"tool"
            text:         输入内容（文本时为字符串）
            keep_history: 1=保留历史（多轮对话），0=不保留
            input_type:   输入类型，默认文本
        输出：本轮模型生成的完整文本（Python 字符串）
        """
        inp = RKLLMInput()
        ctypes.memset(byref(inp), 0, ctypes.sizeof(inp))
        inp.role = role.encode("utf-8")
        inp.enable_thinking = False
        inp.input_type = input_type
        inp.u.prompt_input = text.encode("utf-8")

        infer = RKLLMInferParam()
        ctypes.memset(byref(infer), 0, ctypes.sizeof(infer))
        infer.mode = RKLLM_INFER_GENERATE
        infer.keep_history = keep_history

        self._chunks = []
        ret = self.lib.rkllm_run(self.handle, byref(inp), byref(infer), None)
        if ret != 0:
            raise RuntimeError(f"rkllm_run failed, ret={ret}")
        return "".join(self._chunks)

    def destroy(self):
        """释放模型和 NPU 资源。程序退出前必须调用，否则 NPU 可能不释放。"""
        if self.handle:
            self.lib.rkllm_destroy(self.handle)
            self.handle = None
