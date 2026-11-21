"""Выбор актуальной детали по фактам графа без вызова модели."""

import re

from assistant.infra.grounding import code_type, find_codes


def current_request(question: str) -> bool:
    """Исторический вопрос не требует рекомендовать новую деталь."""
    if re.search(r"раньше|ранее|истори|первоначаль|изначаль|до замены", question, re.IGNORECASE):
        return False
    return bool(re.search(r"заказ|закаж|сейчас|актуаль|замен|постав", question, re.IGNORECASE))


def current_part(question: str, facts: list[str]) -> tuple[str | None, list[str], str | None]:
    """Возвращает конечную деталь, её поставщиков и причину неоднозначности."""
    edges = set()
    for fact in facts:
        match = re.fullmatch(r"(?:\d+[.)]\s+)?(.+?) -([A-Z_]+)-> (.+)", fact.strip())
        if match:
            source, kind, target = match.groups()
            edges.add((source.removeprefix("Part:"), kind, target.removeprefix("Part:")))
    codes = find_codes(question)
    models = {code for code in codes if code_type(code) == "Model"}
    errors = {code for code in codes if code_type(code) == "ErrorCode"}
    parts = {code for code in codes if code_type(code) == "Part"}
    parts &= {node for source, _, target in edges for node in (source, target)}
    components = {target for source, kind, target in edges if kind == "INDICATES" and source in errors}
    components.update(target for _, kind, target in edges
                      if kind == "IS_A" and target.lower() in question.lower())
    if not parts:
        parts = {target for source, kind, target in edges if kind == "HAS_PART" and source in models
                 and any(s == target and k == "IS_A" and t in components for s, k, t in edges)}
    if not parts:
        return None, [], None
    final = set()
    for part in sorted(parts):
        visited = set()
        while True:
            if part in visited:
                return None, [], "В цепочке замен найден цикл. Нельзя выбрать актуальную деталь."
            visited.add(part)
            replacements = {target for source, kind, target in edges if source == part and kind == "REPLACED_BY"}
            if len(replacements) > 1:
                return None, [], "В цепочке несколько замен. Нужны уточняющие данные."
            if not replacements:
                final.add(part)
                break
            part = next(iter(replacements))
    if len(final) != 1:
        return None, [], "Подходят несколько деталей. Нужны уточняющие данные."
    part = next(iter(final))
    suppliers = sorted(target for source, kind, target in edges if source == part and kind == "SUPPLIED_BY")
    return part, suppliers, None
