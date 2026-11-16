"""Сообщества графа и их сводки. Требование FR-3."""

import re
from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if not graph:
        return []
    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        groups = [{node} for node in sorted(undirected)]
    else:
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=7)
    members = sorted(sorted(group) for group in groups)
    return [Community(number, nodes) for number, nodes in enumerate(members)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    facts = []
    failures_by_model = Counter()
    failures_by_component = Counter()
    failures_by_line = Counter()
    models = set()
    for node in community.nodes:
        if graph.nodes[node]["type"] == "Model":
            models.add(node)
        elif graph.nodes[node]["type"] == "Part":
            for source, _, attrs in graph.in_edges(node, data=True):
                if attrs["type"] == "HAS_PART":
                    models.add(source)

    model_lines = {}
    context = []
    for model in sorted(models):
        lines = []
        for _, target, attrs in graph.out_edges(model, data=True):
            if attrs["type"] == "BELONGS_TO":
                lines.append(graph.nodes[target]["name"])
        model_lines[model] = lines or ["линейка не указана"]
        context.append(f"{graph.nodes[model]['name']}: {', '.join(model_lines[model])}")
        for _, part, attrs in graph.out_edges(model, data=True):
            if attrs["type"] == "HAS_PART" and part in community.nodes:
                context.append(f"{graph.nodes[model]['name']} содержит {graph.nodes[part]['name']}")

    # Каждая связь принадлежит группе своего начала, даже если её конец в другой группе.
    for source, target, attrs in sorted(graph.out_edges(community.nodes, data=True)):
        source_name = graph.nodes[source]["name"]
        target_name = graph.nodes[target]["name"]
        fact = f"{source_name} -{attrs['type']}-> {target_name}"
        if attrs["type"] == "FAILED":
            fact += f"; отказов {attrs['weight']}"
        facts.append(fact)
        if attrs["type"] == "FAILED":
            for line in model_lines[source]:
                failures_by_line[line] += attrs["weight"]
                failures_by_model[(line, source_name)] += attrs["weight"]
                failures_by_component[(line, target_name)] += attrs["weight"]

    statistics = []
    for line, count in sorted(failures_by_line.items()):
        statistics.append(f"{line}: всего отказов {count}")
    for (line, model), count in sorted(failures_by_model.items()):
        statistics.append(f"{line}: модель {model}, отказов {count}")
    for (line, component), count in sorted(failures_by_component.items()):
        statistics.append(f"{line}: узел {component}, отказов {count}")

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши сводку сообщества графа знаний не длиннее 150 слов. "
                    "Опиши только основные сущности, связи и замены деталей. "
                    "Статистику в текст не включай: программа добавит её точно после твоего ответа. "
                    "Все числа уже посчитаны кодом: используй их без пересчёта и выдумок. "
                    "Сохрани численность отказов по линейкам, моделям и узлам, если она дана. "
                    "Это статистика части графа, поэтому не объявляй её итогом по всему корпусу. "
                    "Контекст моделей нужен для понимания фактов; не считай его новыми отказами."
                    " Типы сущностей не взаимозаменяемы: код E — ошибка, P — деталь, AR/BZ/GS — модель. "
                    "Статистику бери ТОЛЬКО из раздела готовой статистики. "
                    "Количество деталей, ошибок и других связей не является числом отказов. "
                    "Если готовой статистики нет, явно скажи, что в этом сообществе нет данных об отказах."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Узлы:\n"
                    + "\n".join(sorted(community.nodes))
                    + "\nКонтекст моделей:\n"
                    + "\n".join(context)
                    + "\nФакты:\n"
                    + "\n".join(facts)
                    + "\nСтатистика отказов, посчитанная кодом:\n"
                    + ("\n".join(statistics) or "Нет данных об отказах в этом сообществе. Не придумывай числа.")
                ),
            },
        ]
    )
    # Числовая часть не зависит от того, насколько точно модель пересказала числа.
    # Убираем её попытки повторить статистику; оставляем описание сущностей и связей.
    description = []
    for sentence in re.split(r"[.!?\n]+", response.content):
        if re.search(r"отказ|статистик|(?<![\w-])\d+(?![\w-])", sentence, flags=re.IGNORECASE):
            continue
        if sentence.strip():
            description.append(sentence.strip())
    exact = "Статистика части графа: " + "; ".join(statistics) if statistics else "В этой группе нет данных об отказах"
    return " ".join((exact + ". " + ". ".join(description)).split()[:150])
