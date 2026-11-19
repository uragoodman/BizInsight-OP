"""Ragas 兼容层：修复 langchain-community 0.4+ 的 vertexai 移除问题。

ragas 0.4.3 在 ``ragas/llms/base.py`` 的 ``MULTIPLE_COMPLETION_SUPPORTED``
类型列表里引用 ``ChatVertexAI``/``VertexAI``，但 langchain-community 0.4+
已把 provider 拆分移除（sunset）。实际 judge 走 OpenAI 兼容接口，不会实例化
这两个类——所以注入最小 stub 让 import 通过即可，不改动 site-packages。

任何 import ragas 的入口都应先 ``import app.rag.ragas_compat``。
"""

from __future__ import annotations

import sys
import types


def _install_stubs() -> None:
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        from langchain_community import chat_models

        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = stub
    from langchain_community import llms

    if not hasattr(llms, "VertexAI"):
        llms.VertexAI = type("VertexAI", (), {})


_install_stubs()
