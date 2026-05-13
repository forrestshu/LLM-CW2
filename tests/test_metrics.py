from backend.app.core.metrics import count_arguments, section_diversity, strip_thinking


def test_strip_thinking_removes_hidden_block():
    text = "Visible\n<think>secret reasoning</think>\nDone"
    assert strip_thinking(text) == "Visible\n\nDone"


def test_count_arguments_supports_english_and_chinese():
    english = "Argument One: A\n- Core claim: x\nArgument Two: B\nArgument Three: C"
    chinese = "论点一：甲\n- 核心主张：x\n论点二：乙\n论点三：丙"
    assert count_arguments(english) == 3
    assert count_arguments(chinese) == 3


def test_section_diversity_returns_range():
    score = section_diversity(["abcde", "vwxyz", "abxyz"])
    assert 0 <= score <= 1

