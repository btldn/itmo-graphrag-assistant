"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


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
    return Answer(text, sources=sorted(sources), facts=facts, refused="не найдено" in text.lower())


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partial_answers = []
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
        if set(find_codes(text)) - set(find_codes(question + "\n" + community.summary)):
            continue
        partial_answers.append(text.strip())
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
                    "Если данных недостаточно, напиши: «В базе знаний не найдено.»"
                ),
            },
            {
                "role": "user",
                "content": "Полезные ответы:\n" + "\n\n".join(partial_answers) + "\n\nВопрос: " + question
                + "\nСформулируй ответ, сохранив найденные факты. Не требуй ответов от остальных групп.",
            },
        ]
    )
    text = response.content.strip() or "В базе знаний не найдено."
    if set(find_codes(text)) - set(find_codes(question + "\n" + "\n".join(partial_answers))):
        return Answer("В базе знаний не найдено: ответ содержит неподтверждённый код.",
                      sources=sorted(sources), refused=True)
    return Answer(text, sources=sorted(sources), refused="не найдено" in text.lower())
