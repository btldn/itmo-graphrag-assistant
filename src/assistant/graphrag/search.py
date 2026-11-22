"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
import re
from itertools import combinations, pairwise

import networkx as nx

from assistant.graphrag.parts import current_part, current_request
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

AMOUNT = r"\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?"
ERP_AMOUNT = re.compile(
    rf"\b{AMOUNT}\s*(?:руб\w*|₽)"
    rf"|\b(?:цен[аы]|стоимость|стоит|остаток|остал(?:ось|ись|ся|ась)|в\s+наличии)\s*[:=—-]?\s*{AMOUNT}"
    rf"|\b(?:есть|доступно|на\s+складе[^.!?\n]*?)\s+{AMOUNT}\s*(?:шт\w*|единиц\w*|детал\w*)",
    re.IGNORECASE,
)


def _select_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    """Выбирает связи рядом с одной сущностью или на путях между несколькими."""
    undirected = graph.to_undirected()
    if len(anchors) == 1:
        nearby = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        return set(graph.subgraph(nearby).edges())

    edges = set()
    for start, end in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
            for source, target in pairwise(path):
                # Путь ищем в обе стороны, но сам факт сохраняем с исходным направлением.
                if graph.has_edge(source, target):
                    edges.add((source, target))
                if graph.has_edge(target, source):
                    edges.add((target, source))

    pending = []
    for source, target in edges:
        for node in (source, target):
            if graph.nodes[node]["type"] == "Part":
                pending.append(node)
    visited = set()
    while pending:
        part = pending.pop()
        if part in visited:
            continue
        visited.add(part)
        for source, target, attrs in graph.out_edges(part, data=True):
            if attrs["type"] == "REPLACED_BY":
                edges.add((source, target))
                pending.append(target)
    return edges


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    anchors = []
    for code in find_codes(question):
        node = f"{code_type(code)}:{code}"
        if node not in graph:
            return Answer("В базе знаний не найдено.", refused=True)
        anchors.append(node)
    for kind, name in find_names(question, kb.vocabulary):
        node = f"{kind}:{name}"
        if node in graph and node not in anchors:
            anchors.append(node)
    if not anchors:
        return Answer("В базе знаний не найдено.", refused=True)

    edges = _select_edges(graph, anchors)
    facts = []
    sources = set()
    context_length = 0
    # При ограничении объёма сначала показываем замены, чтобы не рекомендовать старую деталь.
    ordered_edges = sorted(edges, key=lambda edge: (graph.edges[edge]["type"] != "REPLACED_BY", edge))
    for source, target in ordered_edges:
        attrs = graph.edges[source, target]
        fact = f"{graph.nodes[source]['name']} -{attrs['type']}-> {graph.nodes[target]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        if context_length + len(line) + 1 > 8000:
            break
        facts.append(line)
        context_length += len(line) + 1
        for chunk_id in attrs["sources"]:
            sources.add(chunk_id.split("#", 1)[0])
    if not facts:
        return Answer("В базе знаний не найдено.", refused=True)

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ты ассистент сервисной службы. Отвечай только по пронумерованным фактам ниже. "
                    "Не придумывай коды, цены и остатки. Учитывай модель оборудования из вопроса. "
                    "Смысл связей: BELONGS_TO — модель входит в линейку; HAS_PART — в модели установлена деталь; "
                    "IS_A — деталь относится к узлу; INDICATES — ошибка указывает на узел; "
                    "FIXED_BY — ошибка устраняется по регламенту; SUPPLIED_BY — поставщик детали; "
                    "FAILED — исторический отказ узла в модели, а не тип модели и не номер детали. "
                    "Связь «ошибка -FIXED_BY-> регламент» прямо отвечает на вопрос о регламенте. "
                    "Связь «модель -BELONGS_TO-> линейка» прямо отвечает на вопрос о линейке. "
                    "Не выбирай деталь по одному IS_A без привязки HAS_PART к нужной модели. "
                    "Дай короткий прямой ответ, не пересказывай все факты. "
                    "Если есть цепочка REPLACED_BY, называй последнюю деталь цепочки. "
                    "Если ответа нет, напиши: «В базе знаний не найдено.»"
                ),
            },
            {
                "role": "user",
                "content": "Факты:\n" + "\n".join(facts) + "\n\nВопрос: " + question
                + "\nОтветь по подходящим связям выше. Если прямая связь даёт ответ, используй её без отказа.",
            },
        ]
    )
    text = response.content.strip() or "В базе знаний не найдено."
    if set(find_codes(text)) - set(find_codes(question + "\n" + "\n".join(facts))):
        return Answer("В базе знаний не найдено: ответ содержит неподтверждённый код.",
                      sources=sorted(sources), facts=facts, refused=True)
    if ERP_AMOUNT.search(re.sub(r"[*`_]", "", text)):
        return Answer("В базе знаний не найдено цены или остатка. Эти данные нужно проверить в ERP.",
                      sources=sorted(sources), facts=facts, refused=True)
    refused = "не найдено" in text.lower()
    if not refused and current_request(question) and any(" -REPLACED_BY-> " in fact for fact in facts):
        part, _, problem = current_part(question, facts)
        mentioned = {code for code in find_codes(text) if code_type(code) == "Part"}
        recommends_order = re.search(r"зака[зж]|рекоменд|какую\s+деталь", question, re.IGNORECASE)
        if (mentioned or recommends_order) and (problem or (part and part not in mentioned)):
            return Answer("В базе знаний не найдено подтверждения предложенной замены.",
                          sources=sorted(sources), facts=facts, refused=True)
        if part:
            for sentence in re.split(r"[;\n]|(?<=[.!?])\s+", text):
                parts = {code for code in find_codes(sentence) if code_type(code) == "Part"}
                recommendation = re.search(r"закаж|заказать|рекоменд|нужн", sentence, re.IGNORECASE)
                history = re.search(r"снят|ранее|раньше|предшествен|заменена|не\s+.*заказ", sentence, re.IGNORECASE)
                if parts and part not in parts and recommendation and not history:
                    return Answer("В базе знаний не найдено подтверждения предложенной замены.",
                                  sources=sorted(sources), facts=facts, refused=True)
    return Answer(text, sources=sorted(sources), facts=facts, refused=refused)


def _no_data_only(text: str) -> bool:
    """Отбрасывает сообщения только об отсутствии данных, сохраняя смешанные ответы."""
    sentences = [part.strip() for part in re.split(r"[.!?\n]+|,\s+(?:но|однако)\s+", text) if part.strip()]
    return bool(sentences) and all(re.search(
        r"не указан|не найден|не содерж|не связан|нет (?:данных|информации|сведений)"
        r"|недостаточно (?:данных|информации)",
        sentence, re.IGNORECASE,
    ) for sentence in sentences)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    asked_components = {name for kind, name in find_names(question, kb.vocabulary) if kind == "Component"}
    negative_status = r"не\s+снят[аоы]?\s+с\s+производства"
    partial_answers = []
    supporting_summaries = []
    sources = set()
    for community in kb.communities:
        if not community.summary.strip():
            continue
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос только по сводке. Верни JSON по схеме: answer — "
                        "частичный ответ, score — целая оценка полезности от 0 до 10. "
                        "Если данных для вопроса нет, поставь 0. Сохрани относящиеся к вопросу "
                        "коды и числа. Не выдавай сведения одной группы за итог по всему корпусу."
                        " Сохраняй линейку, узел и число отказов вместе. "
                        "По схеме проекта REPLACED_BY означает: старая деталь снята с производства и заменена. "
                        "Если ответ сообщает только об отсутствии данных, score должен быть 0. "
                        "Отсутствие информации не доказывает, что деталь не снята с производства."
                    ),
                },
                {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            data = json.loads(response.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        text, score = data.get("answer"), data.get("score")
        if not isinstance(text, str) or not text.strip() or type(score) is not int or not 0 < score <= 10:
            continue
        if ERP_AMOUNT.search(re.sub(r"[*`_]", "", text)):
            continue
        if set(find_codes(text)) - set(find_codes(question + "\n" + community.summary)):
            continue
        if _no_data_only(text):
            continue
        useful_text = " ".join(sentence for sentence in re.split(r"[.!?\n]+", text) if not _no_data_only(sentence))
        answered_components = {name for kind, name in find_names(useful_text, kb.vocabulary) if kind == "Component"}
        answered_components.update(
            name for name in kb.vocabulary.get("Component", [])
            if all(re.search(rf"\b{re.escape(word)}\w*", useful_text, re.IGNORECASE) for word in name.split())
        )
        if asked_components and answered_components and asked_components.isdisjoint(answered_components):
            continue
        if re.search(negative_status, text, re.IGNORECASE) and not re.search(
            negative_status, community.summary, re.IGNORECASE
        ):
            continue
        partial_answers.append(text.strip())
        supporting_summaries.append(community.summary)
        for node in community.nodes:
            for chunk_id in kb.graph.nodes[node]["sources"]:
                sources.add(chunk_id.split("#", 1)[0])
        for _, _, attrs in kb.graph.out_edges(community.nodes, data=True):
            for chunk_id in attrs["sources"]:
                sources.add(chunk_id.split("#", 1)[0])

    if not partial_answers:
        return Answer("В базе знаний не найдено.", refused=True)

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Составь общий ответ только по частичным ответам ниже. "
                    "Учитывай все относящиеся к вопросу группы, линейки и модели. "
                    "Числа по одной модели из разных пересказов не считай повторно. "
                    "Не придумывай факты, цены и остатки. "
                    "Отсутствие сведений в других группах не отменяет полезный ответ. "
                    "Если передан единственный полезный ответ и он отвечает на вопрос, сохрани его. "
                    "Подтверждающие сводки сохраняют контекст и числа: сверяй с ними краткие ответы. "
                    "Не превращай отсутствие информации в отрицательный факт. "
                    "Связь REPLACED_BY означает снятие старой детали с производства и её замену. "
                    "Ты только ищешь документы: не создаёшь и не подтверждаешь заказы. "
                    "Если данных недостаточно, напиши: «В базе знаний не найдено.»"
                ),
            },
            {
                "role": "user",
                "content": "Полезные ответы:\n" + "\n\n".join(partial_answers) + "\n\nВопрос: " + question
                + "\n\nПодтверждающие сводки:\n" + "\n\n".join(supporting_summaries)
                + "\nСформулируй ответ, сохранив найденные факты. Не требуй ответов от остальных групп.",
            },
        ]
    )
    text = response.content.strip() or "В базе знаний не найдено."
    retired = set()
    if (re.search(r"детал", question, re.IGNORECASE) and re.search(r"снят", question, re.IGNORECASE)
            and not re.search(r"не\s+снят", question, re.IGNORECASE)):
        lines = [name for kind, name in find_names(question, kb.vocabulary) if kind == "ProductLine"]
        for summary in supporting_summaries:
            for line in lines:
                retired.update(re.findall(rf"{re.escape(line)}:\s*(P-\d{{4}}[A-Z]?) снята с производства", summary))
        if retired - set(find_codes(text)):
            # Это точные статусы, добавленные кодом в сводки, а не догадки по score.
            text = "По подтверждающим сводкам сняты с производства: " + ", ".join(sorted(retired)) + "."
    if re.search(r"\bзаказ[^.!?\n]*\b(?:создан|оформлен|подтвержд[её]н)\b", text, re.IGNORECASE):
        return Answer("Поиск по сводкам не создаёт заказы. Для подготовки заказа нужен режим агента.",
                      sources=sorted(sources), refused=True)
    if ERP_AMOUNT.search(re.sub(r"[*`_]", "", text)):
        return Answer("В базе знаний не найдено цены или остатка. Эти данные нужно проверить в ERP.",
                      sources=sorted(sources), refused=True)
    support = "\n".join(partial_answers + supporting_summaries)
    if re.search(negative_status, text, re.IGNORECASE) and not re.search(negative_status, support, re.IGNORECASE):
        return Answer("В базе знаний не найдено подтверждения отрицательного статуса детали.",
                      sources=sorted(sources), refused=True)
    if _no_data_only(text):
        # Полезные сведения не скрываем, но полноту ответа не объявляем доказанной.
        return Answer("В базе знаний не найдено полного подтверждённого ответа. "
                      "Найденные частичные сведения: " + " ".join(partial_answers),
                      sources=sorted(sources), refused=True)
    if set(find_codes(text)) - (set(find_codes(question + "\n" + "\n".join(partial_answers))) | retired):
        return Answer("В базе знаний не найдено: ответ содержит неподтверждённый код.",
                      sources=sorted(sources), refused=True)
    return Answer(text, sources=sorted(sources), refused="не найдено" in text.lower())
