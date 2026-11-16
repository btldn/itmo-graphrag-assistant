"""Небольшой граф с готовыми фактами: python -m assistant.graphrag.demo."""

from html import escape
from pathlib import Path

from assistant.graphrag.graph_builder import build_graph
from assistant.infra.schemas import Entity, ExtractionResult, Relation


def main() -> None:
    """Собирает граф примера и сохраняет автономную HTML-схему."""
    model = Entity("Model", "AR-210")
    error = Entity("ErrorCode", "E-17")
    component = Entity("Component", "пусковое реле")
    old = Entity("Part", "P-1042")
    new = Entity("Part", "P-1042B")
    results = [
        ExtractionResult(
            "manual_AR-210.md#0",
            "manual_AR-210.md",
            [model, old, component],
            [Relation(model, "HAS_PART", old), Relation(old, "IS_A", component)],
        ),
        ExtractionResult(
            "manual_AR-210.md#1",
            "manual_AR-210.md",
            [error, component],
            [Relation(error, "INDICATES", component)],
        ),
        ExtractionResult(
            "bulletin_SB-2025-03.md#0",
            "bulletin_SB-2025-03.md",
            [old, new],
            [Relation(old, "REPLACED_BY", new)],
        ),
    ]
    graph = build_graph(results)
    assert graph.number_of_nodes() == 5 and graph.number_of_edges() == 4
    assert graph.has_edge(old.node_id, new.node_id)

    positions = {
        model.node_id: (160, 350),
        error.node_id: (160, 100),
        component.node_id: (540, 100),
        old.node_id: (540, 350),
        new.node_id: (920, 350),
    }
    labels = {
        "HAS_PART": "содержит",
        "IS_A": "является",
        "INDICATES": "указывает на",
        "REPLACED_BY": "заменена на",
    }
    arrows, nodes, rows = [], [], []
    for source, target, attrs in graph.edges(data=True):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        if x1 == x2:
            direction = 1 if y2 > y1 else -1
            y1 += 42 * direction
            y2 -= 48 * direction
            label_x, label_y = x1 + 65, (y1 + y2) / 2
        else:
            direction = 1 if x2 > x1 else -1
            x1 += 110 * direction
            x2 -= 116 * direction
            label_x, label_y = (x1 + x2) / 2, y1 - 14
        arrows.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)"/>'
            f'<text class="relation" x="{label_x}" y="{label_y}">{escape(labels[attrs["type"]])}</text>'
        )
        rows.append(
            f"<tr><td>{escape(graph.nodes[source]['name'])}</td>"
            f"<td>{escape(attrs['type'])}</td><td>{escape(graph.nodes[target]['name'])}</td>"
            f"<td>{attrs['weight']}</td><td>{escape(', '.join(attrs['sources']))}</td></tr>"
        )
    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        css_class = "node current" if node == new.node_id else "node"
        nodes.append(
            f'<g class="{css_class}"><title>{escape(node)}</title>'
            f'<rect x="{x - 110}" y="{y - 42}" width="220" height="84" rx="16"/>'
            f'<text class="kind" x="{x}" y="{y - 12}">{escape(attrs["type"])}</text>'
            f'<text class="name" x="{x}" y="{y + 16}">{escape(attrs["name"])}</text></g>'
        )

    page = (
        """<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Граф знаний — учебный пример</title>
<style>
body{margin:0;background:#f3f5fa;color:#18243b;font:16px/1.6 system-ui,sans-serif}
main{max-width:1120px;margin:40px auto;padding:0 24px}h1{line-height:1.2;margin:10px 0}
.badge{color:#475569;font-size:14px}.card{background:white;border:1px solid #dce3ee;
border-radius:20px;padding:24px;margin:24px 0}.diagram,.table{overflow-x:auto}
svg{display:block;width:100%;min-width:760px}line{stroke:#64748b;stroke-width:2}
text{text-anchor:middle;fill:#18243b;font-family:system-ui,sans-serif}
.relation{font-size:14px}.node rect{fill:#eff5ff;stroke:#94b4e8;stroke-width:1.5}
.current rect{fill:#dcfce7;stroke:#22a567}.kind{font-size:13px;fill:#52647d}
.name{font-size:18px;font-weight:600}h2{font-size:20px;margin-top:0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;
padding:12px;border-bottom:1px solid #e2e8f0;white-space:nowrap}th{color:#52647d}
</style><main>
<div class="badge">ФИКСИРОВАННЫЙ ПРИМЕР · 5 УЗЛОВ · 4 СВЯЗИ</div>
<h1>Как документы соединяются в граф</h1>
<p>Вопрос сотрудника: «Что заказать при ошибке E-17 на AR-210?»</p>
<section class="card diagram" aria-label="Схема графа знаний">
<svg viewBox="0 0 1100 440" role="img" aria-labelledby="graph-title graph-description">
<title id="graph-title">Модель, ошибка, пусковое реле и замена детали</title>
<desc id="graph-description">AR-210 содержит P-1042. P-1042 является пусковым реле.
E-17 указывает на пусковое реле. P-1042 заменена на P-1042B.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8"
markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>
"""
        + "\n".join(arrows + nodes)
        + """</svg></section>
<section class="card"><h2>Как прочитать схему</h2>
<p>Ошибка указывает на пусковое реле. В модели AR-210 это деталь P-1042.
Бюллетень сообщает о замене на <strong>P-1042B</strong> — она выделена зелёным.</p>
<p>Факты для демонстрации заданы в коде. Граф собран нашей функцией build_graph.
Модель и поиск здесь не запускаются.</p></section>
<section class="card"><h2>Факты и их источники</h2><div class="table"><table>
<thead><tr><th>Откуда</th><th>Связь</th><th>Куда</th><th>Вес</th><th>Источник</th></tr></thead>
<tbody>"""
        + "\n".join(rows)
        + """</tbody></table></div>
<p>Вес — число разных фрагментов, в которых найдена связь.</p></section></main></html>"""
    )
    output = Path("artifacts/demo_graph.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"Граф готов: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} связи.")
    print(f"Открой в браузере: {output.resolve()}")


if __name__ == "__main__":
    main()
