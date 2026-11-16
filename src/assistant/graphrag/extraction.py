"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, RELATION_SIGNATURES, Chunk, Entity, ExtractionResult, Relation


def _parse_response(text: str) -> dict:
    """Проверяет структуру JSON до проверки фактов по тексту."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Ответ должен быть объектом JSON")

    for field in ("entities", "relations"):
        rows = data.get(field)
        if not isinstance(rows, list):
            raise ValueError(f"Поле {field} должно быть списком")
        schema = EXTRACTION_SCHEMA["properties"][field]["items"]
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Элемент списка должен быть объектом")
            for key in schema["required"]:
                value = row.get(key)
                if not isinstance(value, str):
                    raise ValueError(f"Поле {key} должно быть строкой")
                allowed = schema["properties"][key].get("enum")
                if allowed is not None and value not in allowed:
                    raise ValueError(f"Недопустимое значение поля {key}")
    return data


def _checked_entity(kind: str, raw: str, codes: set[str], names: dict[str, list[str]]) -> Entity | None:
    """Принимает код из чанка или название из словаря и приводит его к единому виду."""
    name = canonical_name(kind, raw, names)
    if name is None or (kind in CODE_TYPES and name not in codes):
        return None
    return Entity(kind, name)


def _explicit_relations(text: str, names: dict[str, list[str]]) -> list[Relation]:
    """Явные формулировки корпуса, которые модель часто пропускает."""
    relations = []
    for sentence in re.split(r"[.!?\n]", text):
        codes = find_codes(sentence)
        models = [code for code in codes if code_type(code) == "Model"]
        lines = [name for name in names.get("ProductLine", []) if name.lower() in sentence.lower()]
        if (re.search(r"\bвходит\s+в\s+линейку\b", sentence, flags=re.IGNORECASE)
                and not re.search(r"\bне\s+входит\b", sentence, flags=re.IGNORECASE)):
            if len(models) == 1 and len(lines) == 1:
                relations.append(Relation(Entity("Model", models[0]), "BELONGS_TO", Entity("ProductLine", lines[0])))
        errors = [code for code in codes if code_type(code) == "ErrorCode"]
        procedures = [code for code in codes if code_type(code) == "Procedure"]
        if (re.search(r"\bприменяется\s+при\s+ошибке\b", sentence, flags=re.IGNORECASE)
                and not re.search(r"\bне\s+применяется\b", sentence, flags=re.IGNORECASE)):
            if len(errors) == 1 and len(procedures) == 1:
                relations.append(Relation(
                    Entity("ErrorCode", errors[0]), "FIXED_BY", Entity("Procedure", procedures[0])
                ))
        suppliers = [name for name in names.get("Supplier", []) if name.lower() in sentence.lower()]
        parts = [code for code in codes if code_type(code) == "Part"]
        if (re.match(r"\s*Поставщик\s+детали\s+", sentence, flags=re.IGNORECASE)
                and not re.search(r"\bне\b", sentence, flags=re.IGNORECASE)):
            if len(parts) == 1 and len(suppliers) == 1:
                relations.append(Relation(Entity("Part", parts[0]), "SUPPLIED_BY", Entity("Supplier", suppliers[0])))
    pattern = (r"\bДеталь\s+([^.\n]+?)\s+снята\s+с\s+производства\.\s+"
               r"Вместо\s+не[её]\s+устанавливается\s+деталь\s+([^.\n]+)\.")
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        old_text, new_text = match.groups()
        old = [code for code in find_codes(old_text) if code_type(code) == "Part"]
        new = [code for code in find_codes(new_text) if code_type(code) == "Part"]
        if len(old) == len(new) == 1 and old != new and not re.search(r"\bне\b", old_text, flags=re.IGNORECASE):
            relations.append(Relation(Entity("Part", old[0]), "REPLACED_BY", Entity("Part", new[0])))
    return relations


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    names = vocabulary()
    # Словарь подтверждает допустимость названия, текст — его наличие в документе.
    names = {kind: [name for name in values if name.lower() in chunk.text.lower()] for kind, values in names.items()}
    codes = set(find_codes(chunk.text))
    directions = "\n".join(f"{kind}: {start} -> {end}" for kind, (start, end) in RELATION_SIGNATURES.items())
    example = {
        "entities": [
            {"type": "Model", "name": "AR-450"},
            {"type": "Part", "name": "P-4500"},
            {"type": "Component", "name": "контроллер"},
            {"type": "Supplier", "name": "АвтоматикаСевер"},
        ],
        "relations": [
            {
                "subject_type": "Model",
                "subject": "AR-450",
                "type": "HAS_PART",
                "object_type": "Part",
                "object": "P-4500",
            },
            {
                "subject_type": "Part", "subject": "P-4500", "type": "IS_A",
                "object_type": "Component", "object": "контроллер",
            },
            {
                "subject_type": "Part", "subject": "P-4500", "type": "SUPPLIED_BY",
                "object_type": "Supplier", "object": "АвтоматикаСевер",
            },
        ],
    }
    prompt = (
        "Извлеки сущности и связи только из данного текста. Верни JSON по заданной схеме.\n"
        "Типы сущностей: Model — модель оборудования, Part — деталь, ErrorCode — ошибка, "
        "Procedure — регламент, ProductLine — линейка, Component — узел, Supplier — поставщик.\n"
        "Допустимые связи и их направления:\n" + directions + "\n"
        "BELONGS_TO — входит в линейку; HAS_PART — содержит деталь; IS_A — является узлом; "
        "INDICATES — указывает на узел; FIXED_BY — устраняется по регламенту; "
        "REPLACED_BY — старая деталь заменена новой (направление строго от старой к новой); "
        "SUPPLIED_BY — поставляется поставщиком; FAILED — зафиксирован отказ узла в модели.\n"
        "Для КАЖДОЙ строки состава оборудования извлеки ВСЕ три связи: "
        "модель HAS_PART деталь, деталь IS_A узел, деталь SUPPLIED_BY поставщик. "
        "Модель указана в заголовке или перед списком. Не пропускай HAS_PART. "
        "Перечень оборудования не означает FAILED: эту связь давай только по отчёту об отказе. "
        "BELONGS_TO добавляй только когда линейка явно названа в тексте, не угадывай её по коду модели.\n"
        "Пример состава: В AR-450 установлены: контроллер — деталь P-4500, "
        "поставщик «АвтоматикаСевер». Связи в JSON relations:\n"
        '[{"subject_type":"Model","subject":"AR-450","type":"HAS_PART",'
        '"object_type":"Part","object":"P-4500"},'
        '{"subject_type":"Part","subject":"P-4500","type":"IS_A",'
        '"object_type":"Component","object":"контроллер"},'
        '{"subject_type":"Part","subject":"P-4500","type":"SUPPLIED_BY",'
        '"object_type":"Supplier","object":"АвтоматикаСевер"}]. '
        "В entities перечисли все концы этих связей. Это только пример формы, не факты текущего текста.\n"
        "Приведи коды к виду AR-210, P-1042B, E-17, SP-104. Названия бери из словаря: "
        + json.dumps(names, ensure_ascii=False)
        + "\n"
        "Номера отчётов и бюллетеней не являются сущностями. Не добавляй сведения из примера, "
        "если их нет в тексте. Если фактов нет, верни пустые списки.\n"
        "Пример текста: В AR-450 установлен контроллер — деталь P-4500, поставщик АвтоматикаСевер.\n"
        "Пример ответа: " + json.dumps(example, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                chunk.text
                + "\n\nИзвлеки все подтверждённые связи. Начни список relations со связей HAS_PART: "
                "для каждой детали из состава укажи модель, в которой она установлена. "
                "Затем добавь IS_A, SUPPLIED_BY и остальные связи. "
                "Если состава оборудования нет, не выдумывай HAS_PART. Верни только JSON."
            ),
        },
    ]
    for _ in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
        try:
            data = _parse_response(response.content)
            break
        except (ValueError, TypeError) as error:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": f"Исправь формат ответа: {error}. Верни JSON по схеме."})
    else:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    explicit = _explicit_relations(chunk.text, names)
    for row in data["entities"]:
        entity = _checked_entity(row["type"], row["name"], codes, names)
        if entity is None:
            result.dropped += 1
        elif entity not in result.entities:
            result.entities.append(entity)

    for row in data["relations"]:
        subject = _checked_entity(row["subject_type"], row["subject"], codes, names)
        target = _checked_entity(row["object_type"], row["object"], codes, names)
        if subject is None or target is None:
            result.dropped += 1
            continue
        relation = Relation(subject, row["type"], target)
        reverse_is_explicit = relation.type == "REPLACED_BY" and Relation(target, "REPLACED_BY", subject) in explicit
        if reverse_is_explicit:
            # Коды подтверждены текстом; ошибочно только направление связи.
            for entity in (subject, target):
                if entity not in result.entities:
                    result.entities.append(entity)
        # Связи FAILED извлекаем только из отчётов о выездах.
        if (not relation.matches_schema()
                or (relation.type == "FAILED" and not chunk.doc_id.startswith("report_"))
                or (relation.type == "REPLACED_BY" and subject == target) or reverse_is_explicit):
            result.dropped += 1
            continue
        # Модель может назвать сущность только внутри связи, пропустив список entities.
        for entity in (subject, target):
            if entity not in result.entities:
                result.entities.append(entity)
        if relation not in result.relations:
            result.relations.append(relation)
    for relation in explicit:
        if relation.type in {"REPLACED_BY", "SUPPLIED_BY"}:
            # Для бюллетеня дополняем только связи между уже принятыми кодами.
            code_entities = [end for end in (relation.subject, relation.object) if end.type in CODE_TYPES]
            if any(entity not in result.entities for entity in code_entities):
                continue
        if relation not in result.relations:
            result.relations.append(relation)
        for entity in (relation.subject, relation.object):
            if entity not in result.entities:
                result.entities.append(entity)
    return result
