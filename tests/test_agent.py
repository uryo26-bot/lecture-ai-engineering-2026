"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""
import json

from pydantic_ai import DeferredToolRequests, ToolDenied
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_is_saved_only_after_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)

    first = agent.run_sync(
        "学習記録を保存してください",
        model=TestModel(call_tools=["save_learning_log"]),
    )

    assert isinstance(first.output, DeferredToolRequests)
    assert len(first.output.approvals) == 1
    assert list(tmp_path.iterdir()) == []

    call = first.output.approvals[0]
    call_args = call.args_as_dict()
    deferred_results = first.output.build_results(
        approvals={call.tool_call_id: True}
    )

    final = agent.run_sync(
        None,
        model=TestModel(call_tools=[]),
        message_history=first.all_messages(),
        deferred_tool_results=deferred_results,
    )

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8")) == call_args
    assert isinstance(final.output, str)

def test_save_learning_log_is_not_saved_after_denial(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)

    first = agent.run_sync(
        "学習記録を保存してください",
        model=TestModel(call_tools=["save_learning_log"]),
    )

    assert isinstance(first.output, DeferredToolRequests)
    call = first.output.approvals[0]
    deferred_results = first.output.build_results(
        approvals={
            call.tool_call_id: ToolDenied("利用者が拒否しました")
        }
    )

    agent.run_sync(
        None,
        model=TestModel(call_tools=[]),
        message_history=first.all_messages(),
        deferred_tool_results=deferred_results,
    )

    assert list(tmp_path.iterdir()) == []
