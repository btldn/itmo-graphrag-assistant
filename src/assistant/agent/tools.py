"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_PATTERN = r"^P-[0-9]{4}[A-Z]?$"

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Найти факты по документации: ошибки, детали оборудования и их актуальные замены.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "minLength": 1}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Ответить на обзорный вопрос по сводкам: отказы, линейки и общие закономерности.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "minLength": 1}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить остаток детали на складе и признак снятия с производства.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": PART_PATTERN},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получить цену детали в рублях и признак снятия с производства из ERP.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": {"type": "string", "pattern": PART_PATTERN}},
                "required": ["part_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ детали для модели оборудования. Вызывать только после подтверждения.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "pattern": PART_PATTERN},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string", "minLength": 1},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
                "additionalProperties": False,
            },
        },
    },
]


def _valid_part(part_id: str) -> bool:
    return isinstance(part_id, str) and re.fullmatch(PART_PATTERN, part_id) is not None


def _erp_result(function: Callable[..., dict], **arguments) -> dict:
    """Превращает постоянные ошибки ERP в результат инструмента, не скрывая недоступность."""
    try:
        return function(**arguments)
    except ErpNotFound as error:
        return {"error": f"Данных нет в ERP: {error}"}
    except ErpRejected as error:
        return {"error": f"ERP отклонила запрос: {error}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def part_fits_model(part_id: str, model: str) -> bool:
        """Деталь установлена в модели или достижима по цепочке её замен."""
        model_node = f"Model:{model}"
        if model_node not in kb.graph:
            return False
        pending = [target for _, target, attrs in kb.graph.out_edges(model_node, data=True)
                   if attrs["type"] == "HAS_PART"]
        visited = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == f"Part:{part_id}":
                return True
            pending.extend(target for _, target, attrs in kb.graph.out_edges(node, data=True)
                           if attrs["type"] == "REPLACED_BY")
        return False

    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Вопрос должен быть непустой строкой."}
        answer = local_search(question, kb, llm)
        # Только точные связи графа, а не утверждения из текста модели.
        graph_facts = {
            f"{kb.graph.nodes[source]['name']} -{attrs['type']}-> {kb.graph.nodes[target]['name']}"
            for source, target, attrs in kb.graph.edges(data=True)
        }
        verified_facts = [
            re.sub(r"^\s*\d+[.)]\s+", "", fact) for fact in answer.facts
            if re.sub(r"^\s*\d+[.)]\s+", "", fact) in graph_facts
        ]
        models = [code for code in find_codes(question) if code_type(code) == "Model"]
        parts = [code for code in find_codes(answer.text) if code_type(code) == "Part"]
        if models and any(not any(part_fits_model(part, model) for model in models) for part in parts):
            return {
                "answer": "В базе знаний не найдено: привязка названной детали к модели не подтверждена.",
                "facts": answer.facts, "sources": answer.sources, "refused": True,
                "verified_facts": verified_facts,
            }
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources,
                "refused": answer.refused, "verified_facts": verified_facts}

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Вопрос должен быть непустой строкой."}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        if not _valid_part(part_id):
            return {"error": "Номер детали должен быть вида P-1042 или P-1042B."}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}."}
        return _erp_result(erp.get_stock, part_id=part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        if not _valid_part(part_id):
            return {"error": "Номер детали должен быть вида P-1042 или P-1042B."}
        return _erp_result(erp.get_price, part_id=part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if not _valid_part(part_id):
            return {"error": "Номер детали должен быть вида P-1042 или P-1042B."}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}."}
        if type(qty) is not int or qty <= 0:
            return {"error": "Количество должно быть целым числом больше нуля."}
        if not isinstance(model, str) or not model.strip():
            return {"error": "Модель оборудования должна быть непустой строкой."}
        if not part_fits_model(part_id, model):
            return {"error": "В графе не подтверждено, что эта деталь подходит указанной модели."}
        return _erp_result(erp.create_order, part_id=part_id, warehouse=warehouse, qty=qty, model=model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
