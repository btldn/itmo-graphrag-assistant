"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from decimal import Decimal

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.graphrag.parts import current_part, current_request
from assistant.infra.config import vocabulary
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, find_names, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

NUMBER = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?![\w-])"
ERP_FAILURE = "Данные ERP получить не удалось."
PRICE_WORDS = r"цен|стоим|стоит|стоят|поч[её]м"
STOCK_WORDS = r"остат|налич|склад|сколько\s+(?:шт|штук)|есть\s+ли"


def _route(question: str) -> str | None:
    """Прямые цены/остатки — ERP; конкретные коды/узлы — local; обзоры не ограничиваем."""
    codes = find_codes(question)
    if (codes and all(code_type(code) == "Part" for code in codes)
            and re.search(f"{PRICE_WORDS}|{STOCK_WORDS}", question, re.IGNORECASE)
            and not re.search(r"замен|вместо|постав|регламент|оформи|закаж", question, re.IGNORECASE)):
        return "erp"
    overview = re.search(r"чаще|частот|статист|больше|сравн|обзор|какие|каких|всех", question, re.IGNORECASE)
    named_component = any(kind == "Component" for kind, _ in find_names(question, vocabulary()))
    return "local" if (codes or named_component) and not overview else None


def _numbers(text: str) -> set[Decimal]:
    """Находит числа, не принимая цифры внутри кодов за количество или цену."""
    text = re.sub(
        r"\b[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?\b",
        lambda match: " " if normalize_code(match.group()) else match.group(),
        text,
    )
    return {Decimal(re.sub(r"[ \u00a0\u202f]", "", raw).replace(",", ".")) for raw in re.findall(NUMBER, text)}


def _prices(text: str) -> set[Decimal]:
    text = re.sub(r"[*`_]", "", text)
    matches = re.findall(rf"({NUMBER})\s*(?:руб\w*|₽)", text, flags=re.IGNORECASE)
    matches += re.findall(rf"(?:цена|стоимость|стоит)\s*[:—-]?\s*({NUMBER})", text, flags=re.IGNORECASE)
    return {number for raw in matches for number in _numbers(raw)}


def _quantities(text: str) -> set[Decimal]:
    """Берёт явно названное количество, а не произвольное число вроде срока доставки."""
    text = re.sub(r"[*`_]", "", text)
    matches = re.findall(rf"({NUMBER})\s*(?:шт\w*|единиц\w*|детал\w*)", text, flags=re.IGNORECASE)
    matches += re.findall(
        rf"(?:закаж\w*|заказ|остаток|остал(?:ось|ись|ся|ась)|количеств\w*|в\s+наличии)\s*[:—-]?\s*({NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    return {number for raw in matches for number in _numbers(raw)}


def _argument_error(arguments: dict, schema: dict) -> str | None:
    """Проверяет обязательные поля и значения по уже заданной схеме инструмента."""
    if not isinstance(arguments, dict):
        return "Аргументы должны быть объектом JSON."
    if set(arguments) != set(schema["required"]):
        return "Нужны только следующие аргументы: " + ", ".join(schema["required"])
    for name, rules in schema["properties"].items():
        value = arguments[name]
        if rules["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            return f"{name}: нужна непустая строка."
        if rules["type"] == "integer" and (type(value) is not int or value < rules["minimum"]):
            return f"{name}: нужно целое положительное число."
        if "enum" in rules and value not in rules["enum"]:
            return f"{name}: допустимы только {', '.join(rules['enum'])}."
        if "pattern" in rules and re.fullmatch(rules["pattern"], value) is None:
            return f"{name}: неправильный формат кода."
    return None


def _execute(function: Callable[..., dict], arguments: dict) -> dict:
    """Повторяет только временно недоступную ERP, всего не больше трёх попыток."""
    for _ in range(3):
        try:
            result = function(**arguments)
            return result if isinstance(result, dict) else {"error": "Инструмент должен вернуть словарь."}
        except ErpUnavailable:
            continue
        except TypeError as error:
            return {"error": f"Ошибка аргументов: {error}"}
    return {"error": ERP_FAILURE}


def _step_evidence(step: ToolStep) -> dict | None:
    """Отделяет факты поиска от текста модели; ошибки не подтверждают ничего."""
    result = step.result
    if step.error or not result or result.get("error"):
        return None
    if step.tool == "search_knowledge":
        if "verified_facts" in result:
            # Поле формирует build_tools после сверки с kb.graph.
            facts = result["verified_facts"]
        elif not result.get("refused"):
            # Совместимость с подставными инструментами старого контракта.
            facts = result.get("facts", [])
        else:
            facts = []
        return {"facts": facts} if facts else None
    if result.get("refused"):
        return None
    return {key: value for key, value in result.items() if key != "sources"}


def _facts_from_steps(steps: list[ToolStep]) -> list[str]:
    facts = []
    for step in steps:
        evidence = _step_evidence(step)
        if evidence:
            facts.extend(evidence.get("facts", []))
    return facts


def _warehouses(text: str) -> set[str]:
    patterns = {
        "Москва": r"\bмоскв[аеуы]\b",
        "Казань": r"\bказан(?:ь|и)\b",
        "Новосибирск": r"\bновосибирск(?:е|а)?\b",
        "Санкт-Петербург": r"\bсанкт[- ]петербург(?:е|а)?\b",
    }
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)}


def _numeric_errors(text: str, question: str, steps: list[ToolStep]) -> list[str]:
    """Сверяет простые фразы, списки и таблицы по паре «деталь, склад»."""
    prices = [step.result for step in steps if step.tool == "get_price" and _step_evidence(step)]
    stocks = [step.result for step in steps if step.tool == "check_stock" and _step_evidence(step)]
    orders = [step.result for step in steps if step.tool == "create_service_order" and _step_evidence(step)]
    all_parts = {row["part_id"] for row in prices + stocks + orders if row.get("part_id")}
    fragments = []
    header = {}
    for line in re.sub(r"[*`_]", "", text).splitlines():
        if "|" in line:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            fields = {index: "цена" if re.search(r"цен|стоим", cell, re.IGNORECASE) else "остаток"
                      for index, cell in enumerate(cells)
                      if re.search(r"цен|стоим|остат|колич|кол-во|налич", cell, re.IGNORECASE)}
            if fields:
                header = fields
                continue
            if all(re.fullmatch(r"[: -]+", cell) for cell in cells):
                continue
            if header:
                if max(header) >= len(cells):
                    return ["Не удалось разобрать числовые столбцы таблицы."]
                subject = " ".join(cell for index, cell in enumerate(cells) if index not in header)
                fragments.extend(f"{subject}: {label} {cells[index]}" for index, label in header.items())
                continue
        else:
            header = {}
        fragments.extend(re.split(r";|(?<=[.!?])\s+|,\s*(?=(?:P-\d{4}|(?:в\s+)?(?:Москв|Казан|Санкт|Новосибирск)))",
                                  line, flags=re.IGNORECASE))
    part = None
    warehouse = None
    facts = _facts_from_steps(steps)
    replacement, _, _ = current_part(question, facts) if current_request(question) else (None, [], None)
    for fragment in fragments:
        price_values = _prices(fragment)
        qty_values = _quantities(fragment)
        has_numbers = bool(price_values or qty_values)
        named = {code for code in find_codes(fragment) if code_type(code) == "Part"}
        if named:
            new_part = next(iter(named)) if len(named) == 1 else None
            if len(named) > 1 and replacement in named and not has_numbers:
                # Заголовок может объяснять замену: «P-...B вместо P-...», без числовых утверждений.
                new_part = replacement
            if (part == replacement and not has_numbers
                    and re.search(r"замен|предшеств|раньше|ранее", fragment, re.IGNORECASE)):
                # Пояснение «является заменой для старой P-...» не меняет заголовок списка.
                new_part = replacement
            if new_part != part:
                warehouse = None
            part = new_part
        places = _warehouses(fragment)
        if places:
            warehouse = next(iter(places)) if len(places) == 1 else None
        if not has_numbers:
            continue
        if len(named) > 1 or len(places) > 1:
            return ["Неоднозначная привязка чисел: укажи каждую деталь и склад отдельной строкой."]
        selected_part = part or (next(iter(all_parts)) if len(all_parts) == 1 else None)
        if selected_part is None:
            return ["Неясно, к какой детали относятся цена или остаток. "
                    "Укажи код детали в каждой строке ответа/таблицы."]
        if price_values:
            confirmed = {number for row in prices if row.get("part_id") == selected_part
                         for number in _numbers(str(row.get("price_rub", "")))}
            if price_values - confirmed:
                return ["Цена не подтверждена для указанной детали."]
        if qty_values:
            ordering = re.search(r"закаж|заказ|нужно|требуется", fragment, re.IGNORECASE)
            stock_claim = re.search(r"остат|остал|налич|есть", fragment, re.IGNORECASE)
            if ordering and not stock_claim:
                if qty_values - _quantities(question):
                    return ["Количество заказа не подтверждено вопросом."]
                continue
            rows = [row for row in stocks if row.get("part_id") == selected_part]
            known_places = {row.get("warehouse") for row in rows}
            selected_place = warehouse or (next(iter(known_places)) if len(known_places) == 1 else None)
            if selected_place is None and len(known_places) > 1:
                return ["Неясно, к какому складу относится остаток."]
            confirmed = {number for row in rows if row.get("warehouse") == selected_place
                         for number in _numbers(str(row.get("qty", "")))}
            if qty_values - confirmed:
                return ["Остаток не подтверждён для указанной детали и склада."]
    return []


def _answer_errors(text: str, question: str, steps: list[ToolStep]) -> list[str]:
    """Сверяет коды и числа с вопросом и результатами, исключая ошибки и источники."""
    text = re.sub(r"[*`_]", "", text)
    results = []
    for step in steps:
        result = _step_evidence(step)
        if result:
            # Номер строки — оформление, а не числовой факт. Сами результаты шагов не меняем.
            if isinstance(result.get("answer"), str):
                result["answer"] = re.sub(r"(?m)^\s*\d+[.)]\s+", "", result["answer"])
            if isinstance(result.get("facts"), list):
                result["facts"] = [re.sub(r"(?m)^\s*\d+[.)]\s+", "", fact) for fact in result["facts"]]
            results.append(result)
    evidence = question + "\n" + json.dumps(results, ensure_ascii=False)
    errors = []
    if set(find_codes(text)) - set(find_codes(evidence)):
        errors.append("В ответе есть неподтверждённый код.")
    if _numbers(text) - _numbers(evidence):
        errors.append("В ответе есть неподтверждённое число.")
    prices = _prices(question)
    quantities = _quantities(question)
    for result in results:
        prices.update(_numbers(str(result.get("price_rub", ""))))
        quantities.update(_numbers(str(result.get("qty", ""))))
    if _prices(text) - prices:
        errors.append("Цена не подтверждена данными о цене.")
    if _quantities(text) - quantities:
        errors.append("Количество не подтверждено данными о количестве.")
    examples = re.findall(rf"(?:например|к примеру)\s*[:,(]?\s*({NUMBER})", text, re.IGNORECASE)
    if any(_numbers(example) - _numbers(question) for example in examples):
        errors.append("Не добавляй числовые примеры, которых пользователь не называл.")
    if re.search(r"какие.*детал", question, re.IGNORECASE) and re.search(r"снят", question, re.IGNORECASE):
        listed = {code for step in steps if step.tool == "summarize_topic" and not step.error
                  and step.result and not step.result.get("refused")
                  for code in find_codes(step.result.get("answer", "")) if code_type(code) == "Part"}
        if listed - set(find_codes(text)) and "не найдено" not in text.lower():
            errors.append("Сохрани весь перечень деталей из ответа summarize_topic, включая промежуточные замены.")
    if current_request(question) and "не найдено" not in text.lower():
        facts = [fact for result in results for fact in result.get("facts", [])]
        part, suppliers, problem = current_part(question, facts)
        if problem:
            errors.append(problem)
        elif part:
            mentioned = {code for code in find_codes(text) if code_type(code) == "Part"}
            if mentioned and part not in mentioned:
                errors.append(f"Ответ должен назвать актуальную деталь {part}, а не только старую.")
            for sentence in re.split(r"[;\n]|(?<=[.!?])\s+", text):
                named = {code for code in find_codes(sentence) if code_type(code) == "Part"}
                recommendation = re.search(r"закаж|заказ|рекоменд|нужн", sentence, re.IGNORECASE)
                history = re.search(r"снят|ранее|раньше|предшествен|заменена|не\s+.*заказ", sentence, re.IGNORECASE)
                if named and part not in named and recommendation and not history:
                    errors.append(f"Нельзя рекомендовать старую деталь вместо {part}.")
                    break
            if re.search(r"постав", question, re.IGNORECASE) and not any(
                name.lower() in text.lower() for name in suppliers
            ):
                errors.append(f"Поставщик актуальной детали {part} не подтверждён.")
    errors.extend(_numeric_errors(text, question, steps))
    if set(re.findall(r"\bSO-\d+\b", text)) - set(re.findall(r"\bSO-\d+\b", evidence)):
        errors.append("Номер заказа не подтверждён.")
    created = any(step.tool == "create_service_order" and step.error is None
                  and step.result and step.result.get("order_id") for step in steps)
    if not created:
        for sentence in re.split(r"[.!?\n]", text.lower()):
            if "заказ" not in sentence:
                continue
            status = r"(?:создан[аоы]?|оформлен[аоы]?|подтвержд[её]н[аоы]?)\b"
            for match in re.finditer(rf"\b{status}", sentence):
                # Проверяем каждое утверждение отдельно: будущее или отрицание
                # одного статуса не оправдывает другой уже выполненный статус.
                prefix = sentence[:match.start()]
                not_completed = re.search(
                    r"\b(?:не|будет|будут|может быть|могут быть|был бы|была бы|было бы|были бы)"
                    r"\s+(?:заказ\w*\s+)?$", prefix,
                )
                if not not_completed:
                    errors.append("Статус заказа не подтверждён: инструмент создания заказа не выполнялся успешно.")
                    break
            else:
                continue
            if errors:
                break
    return errors


def _finish(text: str, question: str, steps: list[ToolStep], messages: list[dict], llm: LLMClient) -> AgentResult:
    """Одна попытка исправления неподтверждённого ответа, затем честный отказ."""
    sources = set()
    order_ids = set()
    for step in steps:
        if step.error is None and step.result:
            sources.update(step.result.get("sources", []))
            if step.tool == "create_service_order" and step.result.get("order_id"):
                order_ids.add(step.result["order_id"])
    for attempt in range(2):
        text = text.strip() or "В базе знаний не найдено."
        for order_id in sorted(order_ids):
            if order_id not in text:
                text += f"\nЗаказ {order_id} создан."
        if any(step.error == ERP_FAILURE for step in steps) and ERP_FAILURE.lower() not in text.lower():
            text += "\n" + ERP_FAILURE
        errors = _answer_errors(text, question, steps)
        if not errors:
            has_data = any(_step_evidence(step) for step in steps)
            refused = "не найдено" in text.lower() or not has_data
            return AgentResult(text, steps=steps, sources=sorted(sources), refused=refused)
        if attempt == 0:
            messages.append({"role": "user", "content": "Исправь ответ по данным инструментов. " + " ".join(errors)
                             + " Не добавляй числовых примеров и номеров строк фактов. "
                             "Если нужно количество заказа, спроси без примера с цифрами."})
            response = llm.chat(messages)
            messages.append(response.as_message())
            text = response.content
    text = "В базе знаний не найдено: не удалось подтвердить ответ."
    if order_ids:
        text += "\nСозданные заказы: " + ", ".join(sorted(order_ids)) + "."
    if any(step.error == ERP_FAILURE for step in steps):
        text += "\n" + ERP_FAILURE
    return AgentResult(text, steps=steps, sources=sorted(sources), refused=True)


def _missing_erp_checks(question: str, steps: list[ToolStep]) -> list[str]:
    checks = []
    if re.search(PRICE_WORDS, question, re.IGNORECASE):
        checks.append(("get_price", None))
    if re.search(STOCK_WORDS, question, re.IGNORECASE):
        for warehouse in _warehouses(question) or {None}:
            checks.append(("check_stock", warehouse))

    missing = []
    for code in find_codes(question):
        for tool, warehouse in checks:
            done = any(
                step.tool == tool
                and step.args.get("part_id") == code
                and (warehouse is None or step.args.get("warehouse") == warehouse)
                and step.result
                and (not step.error or "ERP" in step.error)
                for step in steps
            )
            if not done:
                label = f"{tool} для {code}"
                if warehouse:
                    label += f", склад {warehouse}"
                missing.append(label)
    return missing


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS}
    messages = [
        {
            "role": "system",
            "content": (
                "Ты ассистент сервисной службы. Используй инструменты для документации, цен и остатков. "
                "Сначала найди актуальную деталь через search_knowledge, если её номер неизвестен. "
                "Обзорные вопросы передавай summarize_topic. Не придумывай коды, цены, остатки и количество. "
                "Для заказа количество и модель оборудования должен назвать пользователь. "
                "Перед созданием заказа требуется подтверждение человека. "
                "Различай отсутствие данных, нулевой остаток и недоступность ERP. "
                "Если сведений нет, напиши: «В базе знаний не найдено.»"
                " На конкретный вопрос по документам сначала вызови search_knowledge; "
                "summarize_topic используй для сравнений и обзоров по нескольким отчётам. "
                "Не отказывай без проверки подходящим инструментом. "
                "Подбирай деталь только с привязкой HAS_PART к указанной модели и с учётом замен. "
                "Для просьбы оформить заказ вызови create_service_order: программа сама запросит подтверждение. "
                "Не заявляй, что заказ создан, оформлен или подтверждён, по одному получению цены и остатка."
                " В поисковых подзапросах сохраняй цель исходного вопроса: подбор актуальной детали, "
                "поставщик, цена или подготовка заказа, а не только значение ошибки. "
                "verified_facts — проверенные связи графа; используй их даже при refused=True у текста поиска. "
                "Отсутствие регламента в выбранных фактах не доказывает его отсутствие во всей базе."
            ),
        },
        {"role": "user", "content": question},
    ]
    steps = []
    cache = {}
    known_codes = set(find_codes(question))
    order_reminder_sent = False
    lookup_reminder_sent = False
    replacement_reminder_sent = False
    price_reminder_sent = False
    supplier_reminder_sent = False
    current_reminder_sent = False
    route = _route(question)
    specific_question = route in ("local", "erp")
    if route == "erp":
        messages[0]["content"] += (
            " Это прямой запрос ERP: проверь каждую названную деталь через get_price/check_stock. "
            "В таблице указывай код детали и склад для остатка."
        )
    elif route == "local" and re.search(r"регламент|процедур", question, re.IGNORECASE):
        messages[0]["content"] += (
            " Нужен именно регламент, а не замена детали: вызови search_knowledge с исходным вопросом целиком."
        )
    while len(steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            if (route == "erp" and not lookup_reminder_sent
                    and not any(step.error and step.tool not in ("search_knowledge", "summarize_topic")
                                for step in steps)):
                missing = _missing_erp_checks(question, steps)
                if missing:
                    lookup_reminder_sent = True
                    messages.append({"role": "user", "content": "Прямой запрос ERP ещё не выполнен: "
                                     + "; ".join(missing) + ". Не ищи эти цены/остатки в документах. "
                                     "Если склад не указан, спроси его, не выбирай сам."})
                    continue
            facts = _facts_from_steps(steps)
            part, suppliers, problem = current_part(question, facts)
            mentioned = {code for code in find_codes(response.content) if code_type(code) == "Part"}
            if current_request(question) and part and mentioned and part not in mentioned and not current_reminder_sent:
                current_reminder_sent = True
                messages.append({"role": "user", "content": (
                    f"По полученным связям последняя замена — {part}. Сохрани цель исходного вопроса. "
                    "Для актуального подбора называй эту деталь; старую можно упомянуть только как предшественницу."
                )})
                continue
            asks_supplier = re.search(r"постав", question, flags=re.IGNORECASE)
            supplier_found = any(step.error is None and step.result and not step.result.get("refused")
                                 and any("SUPPLIED_BY" in fact for fact in step.result.get("facts", []))
                                 for step in steps)
            if current_request(question) and (part or problem):
                supplier_found = bool(suppliers) and not problem
            if asks_supplier and steps and not supplier_found and not supplier_reminder_sent:
                supplier_reminder_sent = True
                messages.append({"role": "user", "content": (
                    "Поставщик ещё не найден. Сначала установи деталь нужной модели и последнюю замену, "
                    "затем вызови search_knowledge с конкретным номером актуальной детали и вопросом о поставщике. "
                    "Повторяй только номера из исходного вопроса или полученных фактов; не придумывай их. "
                    + (f"Актуальная деталь — {part}. Ищи её поставщика по этому коду." if part else "")
                )})
                continue
            checked_parts = [step.result["part_id"] for step in steps
                             if step.tool == "check_stock" and step.error is None and step.result
                             and step.result.get("discontinued") is False and step.result.get("part_id")]
            priced_parts = {step.result.get("part_id") for step in steps
                            if step.tool == "get_price" and step.error is None and step.result}
            asks_price = re.search(r"цен|стоим|стоит|стоят", question, flags=re.IGNORECASE)
            if asks_price and checked_parts and checked_parts[-1] not in priced_parts and not price_reminder_sent:
                price_reminder_sent = True
                messages.append({"role": "user", "content": (
                    f"В исходном вопросе запрошена цена, но для проверенной детали {checked_parts[-1]} "
                    "вызов get_price ещё не выполнен. Получи цену инструментом, затем ответь на исходный вопрос."
                )})
                continue
            discontinued = any(step.error is None and step.result and step.result.get("discontinued") is True
                               for step in steps)
            current_part_checked = any(step.error is None and step.result
                                       and step.result.get("discontinued") is False for step in steps)
            selecting_part = re.search(r"заказ|закаж", question, flags=re.IGNORECASE)
            if selecting_part and discontinued and not current_part_checked and not replacement_reminder_sent:
                replacement_reminder_sent = True
                messages.append({"role": "user", "content": (
                    "Проверенная деталь снята с производства. Для подбора детали на заказ нужно "
                    "через search_knowledge найти последнюю замену и проверить её остаток/цену, "
                    "если они запрошены. Не считай discontinued признаком отсутствия замены. "
                    "Если замена не найдена, явно сообщи об этом."
                )})
                continue
            wants_order = re.match(r"\s*(?:оформи(?:те)?|закажи(?:те)?)\b", question, flags=re.IGNORECASE)
            order_attempted = any(step.tool == "create_service_order" for step in steps)
            if specific_question and not wants_order and not steps and not lookup_reminder_sent:
                lookup_reminder_sent = True
                messages.append({"role": "user", "content": (
                    "Ответ ещё не проверен. Вызови search_knowledge для конкретного вопроса по документам "
                    "или check_stock/get_price для прямого запроса остатка/цены известной детали. "
                    "Не отвечай отказом до проверки инструментом."
                )})
                continue
            if wants_order and _quantities(question) and not order_reminder_sent and not order_attempted:
                order_reminder_sent = True
                messages.append({"role": "user", "content": (
                    "Нужно подготовить действие заказа, а не только написать просьбу о подтверждении. "
                    "Если найдена подходящая актуальная деталь, вызови create_service_order с количеством, "
                    "моделью и складом из исходного вопроса. Программа остановит действие для подтверждения. "
                    "Если данных недостаточно, объясни, чего не хватает."
                )})
                continue
            return _finish(response.content, question, steps, messages, llm)

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break
            arguments = call.arguments
            step = ToolStep(call.name, arguments if isinstance(arguments, dict) else {"_raw": arguments})
            steps.append(step)
            if call.name not in schemas or call.name not in tools:
                error = "Неизвестный инструмент. Выбери один из описанных инструментов."
            else:
                error = _argument_error(arguments, schemas[call.name])
            if error is None and call.name == "summarize_topic" and route == "local":
                error = "Вопрос о конкретном коде: используй search_knowledge, а не обзор summarize_topic."
            if error is None and route == "erp" and call.name in ("search_knowledge", "summarize_topic"):
                error = "Прямой запрос ERP: используй get_price/check_stock для каждой названной детали."
            if error is None and "part_id" in arguments and arguments["part_id"] not in known_codes:
                error = "Номер детали не подтверждён. Сначала вызови search_knowledge."
            if error is None and "model" in arguments and set(find_codes(arguments["model"])) - known_codes:
                error = "Код модели не подтверждён вопросом или результатами поиска."
            if error is None and call.name == "create_service_order":
                model = arguments["model"]
                if re.fullmatch(r"(?:AR|BZ|GS)-[0-9]{3}", model) is None or model not in find_codes(question):
                    error = "Для заказа пользователь должен указать код модели оборудования в исходном вопросе."
                facts = _facts_from_steps(steps[:-1])
                part, _, problem = current_part(question, facts)
                if problem:
                    error = problem
                elif part and arguments["part_id"] != part:
                    error = f"Для нового заказа нужна актуальная деталь {part}."
            if error:
                step.error = error
                step.result = {"error": error}
                messages.append(tool_message(call, step.result))
                continue

            if call.name == "create_service_order":
                if Decimal(arguments["qty"]) not in _quantities(question):
                    step.error = "Пользователь не указал такое количество."
                    return AgentResult("Сколько штук заказать?", steps=steps, sources=_sources(steps))
                if not confirm:
                    step.error = "Требуется подтверждение заказа."
                    return AgentResult(
                        f"Подтвердите заказ: {arguments['qty']} шт. {arguments['part_id']} "
                        f"для {arguments['model']}, склад {arguments['warehouse']}.",
                        steps=steps,
                        sources=_sources(steps),
                        needs_confirmation=True,
                        pending_action={"tool": call.name, "args": dict(arguments)},
                    )

            key = (call.name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
            if key not in cache:
                cache[key] = _execute(tools[call.name], arguments)
            step.result = cache[key]
            step.error = step.result.get("error")
            messages.append(tool_message(call, step.result))
            evidence = _step_evidence(step)
            if evidence:
                known_codes.update(find_codes(json.dumps(evidence, ensure_ascii=False)))

    learned = []
    erp_facts = []
    part, _, _ = current_part(question, _facts_from_steps(steps)) if current_request(question) else (None, [], None)
    for step in steps:
        if step.error or not step.result:
            continue
        result = step.result
        if step.tool == "search_knowledge" and result.get("refused"):
            evidence = _step_evidence(step)
            if evidence:
                learned.extend(evidence["facts"])
        elif result.get("answer"):
            learned.append(result["answer"])
        if part and result.get("part_id") != part:
            continue
        if step.tool == "get_price" and "price_rub" in result:
            erp_facts.append(f"{result['part_id']}: цена {result['price_rub']} руб.")
        if step.tool == "check_stock" and "qty" in result:
            erp_facts.append(f"{result['part_id']}: на складе {result['warehouse']} {result['qty']} шт.")
    if part and erp_facts:
        learned = [f"Актуальная деталь: {part}."]
    text = "\n".join(dict.fromkeys(learned + erp_facts))
    result = _finish(text, question, steps, messages, llm)
    missing = [fact for fact in dict.fromkeys(erp_facts) if fact not in result.answer.splitlines()]
    if missing:
        result.answer += "\n" + "\n".join(missing)
    # Факт остановки определяет код, поэтому модель не может убрать его при коррекции.
    result.answer += "\nРабота прервана: достигнут лимит вызовов инструментов."
    return result


def _sources(steps: list[ToolStep]) -> list[str]:
    sources = set()
    for step in steps:
        if step.error is None and step.result:
            sources.update(step.result.get("sources", []))
    return sorted(sources)
