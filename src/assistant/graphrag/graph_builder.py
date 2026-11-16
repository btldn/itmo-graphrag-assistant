"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        relations = [relation for relation in result.relations if relation.matches_schema()]
        entities = list(result.entities)
        for relation in relations:
            entities.extend((relation.subject, relation.object))
        for entity in entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
            sources = graph.nodes[entity.node_id]["sources"]
            if result.chunk_id not in sources:
                sources.append(result.chunk_id)

        for relation in relations:
            source, target = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, weight=0, sources=[])
            edge = graph.edges[source, target]
            if result.chunk_id not in edge["sources"]:
                edge["sources"].append(result.chunk_id)
                edge["weight"] += 1

    # Копия списка позволяет удалять рёбра во время обхода.
    for source, target in list(graph.edges):
        if source == target or not graph.has_edge(source, target) or not graph.has_edge(target, source):
            continue
        forward = graph.edges[source, target]
        backward = graph.edges[target, source]
        if forward["type"] != "REPLACED_BY" or backward["type"] != "REPLACED_BY":
            continue
        graph.graph["conflicts"] += 1
        if forward["weight"] <= backward["weight"]:
            graph.remove_edge(source, target)
        if backward["weight"] <= forward["weight"]:
            graph.remove_edge(target, source)
    return graph
