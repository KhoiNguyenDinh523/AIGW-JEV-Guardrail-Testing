from .base import Guard, GuardResult


def build_guard(name: str, **kw) -> Guard:
    if name == "h2":
        from .h2_llm_guard import LLMGuard
        return LLMGuard(**kw)
    if name == "h3":
        from .h3_jev import JEV
        return JEV(**kw)
    raise ValueError(f"guard không hợp lệ: {name}")
