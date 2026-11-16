"""Регрессии по ошибкам первого прогона настоящей Qwen."""

import json

from assistant.agent.agent import _answer_errors, run_agent
from assistant.agent.tools import build_tools
from assistant.graphrag.communities import summarize_community
from assistant.graphrag.extraction import _explicit_relations, extract_chunk
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import Chunk, Community, ToolStep
from assistant.infra.testing import FakeLLM
from tests.contract.test_tools_and_agent import ORDER, Tools, call


def test_dictionary_name_absent_from_document_does_not_create_a_relation():
    response = json.dumps({
        "entities": [{"type": "Model", "name": "GS-510"}, {"type": "ProductLine", "name": "Арктика"}],
        "relations": [{"subject_type": "Model", "subject": "GS-510", "type": "BELONGS_TO",
                       "object_type": "ProductLine", "object": "Арктика"}],
    })
    result = extract_chunk(Chunk("report#0", "report", "Чиллер gs-510. Отказ насоса."), FakeLLM([response]))
    assert not result.relations
    assert result.dropped == 2


def test_real_part_for_another_model_is_rejected_before_erp(kb, erp):
    tools = build_tools(kb, FakeLLM([]), erp)
    result = tools["create_service_order"](part_id="P-1142", warehouse="Казань", qty=2, model="AR-210")
    assert "error" in result
    assert erp.calls == []


def test_manual_error_table_is_not_a_record_of_actual_failure():
    response = json.dumps({"entities": [], "relations": [
        {"subject_type": "Model", "subject": "AR-450", "type": "FAILED",
         "object_type": "Component", "object": "контроллер"},
    ]})
    result = extract_chunk(Chunk("manual#0", "manual_AR-450.md", "AR-450: отказ узла контроллер."),
                           FakeLLM([response]))
    assert not result.relations and result.dropped == 1


def test_part_cannot_replace_itself():
    response = json.dumps({"entities": [], "relations": [
        {"subject_type": "Part", "subject": "P-4500", "type": "REPLACED_BY",
         "object_type": "Part", "object": "P-4500"},
    ]})
    result = extract_chunk(Chunk("bulletin#0", "bulletin.md", "Замена детали P-4500."), FakeLLM([response]))
    assert not result.relations and result.dropped == 1


def test_search_does_not_confirm_a_part_for_another_model(kb, erp):
    tools = build_tools(kb, FakeLLM(["Для AR-210 закажите P-1142."]), erp)
    result = tools["search_knowledge"]("Какую деталь заказать при E-17 на AR-210?")
    assert result["refused"]
    assert "P-1142" not in result["answer"]


def test_price_or_stock_does_not_prove_that_an_order_was_confirmed():
    tools = Tools()
    llm = FakeLLM([
        call("check_stock", part_id="P-1042B", warehouse="Казань"),
        "Заказ на P-1042B подтверждён.",
        "Заказ на P-1042B оформлен.",
    ])
    result = run_agent("Проверь P-1042B в Казани", llm, tools.as_dict())
    assert result.refused
    assert tools.count("create_service_order") == 0
    assert "подтверждён" not in result.answer and "оформлен" not in result.answer


def test_honest_statement_that_order_was_not_created_is_allowed():
    tools = Tools()
    llm = FakeLLM([
        call("check_stock", part_id="P-1042B", warehouse="Казань"),
        "На складе 3 шт. Заказ не создан.",
    ])
    result = run_agent("Проверь P-1042B в Казани", llm, tools.as_dict())
    assert result.answer == "На складе 3 шт. Заказ не создан."
    assert len(llm.requests) == 2


def test_request_for_order_details_is_not_a_claim_that_order_exists():
    tools = Tools()
    text = "На складе 3 шт. Для оформления заказа укажите количество и подтвердите запрос."
    llm = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Казань"), text])
    result = run_agent("Проверь P-1042B в Казани", llm, tools.as_dict())
    assert result.answer == text and not result.refused
    assert len(llm.requests) == 2


def test_local_does_not_return_a_correct_part_with_an_invented_model(kb):
    answer = local_search("Какую деталь заказать при E-17 на AR-210?", kb,
                          FakeLLM(["Закажите P-1042B. У модели AR-999 тоже стоит эта деталь."]))
    assert answer.refused and "AR-999" not in answer.text


def test_global_discards_a_map_answer_with_an_invented_code(kb):
    llm = FakeLLM(['{"answer":"Модель AR-999","score":10}', '{"answer":"нет","score":0}'])
    answer = global_search("Какая модель чаще отказывает?", kb, llm)
    assert answer.refused and len(llm.requests) == 2


def test_agent_reminds_about_pending_action_once_without_creating_order():
    tools = Tools()
    llm = FakeLLM(["Подтвердите заказ.", call("create_service_order", **ORDER)])
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, tools.as_dict())
    assert result.needs_confirmation
    assert result.pending_action == {"tool": "create_service_order", "args": ORDER}
    assert tools.count("create_service_order") == 0


def test_order_reminder_cannot_produce_an_endless_model_loop():
    llm = FakeLLM(["В базе знаний не найдено.", "В базе знаний не найдено."])
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, Tools().as_dict())
    assert result.refused and len(llm.requests) == 2


def test_explicit_phrases_use_codes_from_text_without_model_guesses():
    names = {"ProductLine": ["Арктика", "Бриз"]}
    result = _explicit_relations(
        "AR-450 входит в линейку «Арктика». Регламент SP-140 применяется при ошибке E-40.", names
    )
    assert {(r.subject.name, r.type, r.object.name) for r in result} == {
        ("AR-450", "BELONGS_TO", "Арктика"), ("E-40", "FIXED_BY", "SP-140"),
    }


def test_explicit_phrases_do_not_use_negation_or_ambiguous_models():
    names = {"ProductLine": ["Арктика"]}
    assert not _explicit_relations("AR-450 не  входит в линейку Арктика.", names)
    assert not _explicit_relations("AR-450 и AR-460 входят в линейку Арктика.", names)
    assert not _explicit_relations("Регламент SP-140 не применяется при ошибке E-40.", names)
    assert not _explicit_relations("Чиллер GS-510. Линейка не указана.", names)


def test_explicit_relations_restore_a_model_omission_and_keep_chunk_sources():
    llm = FakeLLM(['{"entities":[],"relations":[]}'])
    result = extract_chunk(Chunk("manual_AR-450.md#0", "manual_AR-450.md",
                                 "AR-450 входит в линейку Арктика."), llm)
    assert len(llm.requests) == 1
    assert result.chunk_id == "manual_AR-450.md#0"
    assert result.relations[0].type == "BELONGS_TO"
    assert {e.name for e in result.entities} == {"AR-450", "Арктика"}


def test_summary_uses_code_statistics_instead_of_hallucinated_numbers(graph):
    llm = FakeLLM(["Пусковое реле связано с ошибкой. Статистика: всего 999 отказов."])
    text = summarize_community(Community(0, sorted(graph.nodes)), graph, llm)
    assert "37" in text and "999" not in text
    assert "Пусковое реле связано с ошибкой" in text
    assert len(llm.requests) == 1 and len(text.split()) <= 150


def test_specific_question_rejects_overview_then_uses_local_search():
    tools = Tools()
    llm = FakeLLM([
        call("summarize_topic", question="Чем заменена P-1042?"),
        call("search_knowledge", question="Чем заменена P-1042?"),
        "Деталь заменена на P-1042B.",
    ])
    result = run_agent("Чем заменена P-1042?", llm, tools.as_dict())
    assert result.steps[0].error
    assert tools.count("summarize_topic") == 0 and tools.count("search_knowledge") == 1
    assert result.answer == "Деталь заменена на P-1042B."


def test_initial_refusal_gets_one_lookup_reminder():
    llm = FakeLLM(["В базе знаний не найдено.", "В базе знаний не найдено."])
    result = run_agent("Кто поставляет P-3062C?", llm, Tools().as_dict())
    assert result.refused and len(llm.requests) == 2


def test_discontinued_part_does_not_end_selection_before_replacement_lookup():
    tools = Tools(check_stock={"part_id": "P-1042", "qty": 0, "discontinued": True})
    llm = FakeLLM([
        call("check_stock", part_id="P-1042", warehouse="Казань"),
        "P-1042 снята с производства.",
        call("search_knowledge", question="Чем заменена P-1042?"),
        "Нужна P-1042B.",
    ])
    result = run_agent("Какую деталь заказать вместо P-1042? Склад Казань.", llm, tools.as_dict())
    assert tools.count("search_knowledge") == 1
    assert result.answer == "Нужна P-1042B."


def test_price_requested_by_user_is_not_replaced_by_stock_only():
    tools = Tools()
    llm = FakeLLM([
        call("check_stock", part_id="P-1042B", warehouse="Казань"),
        "В Казани есть 3 шт.",
        call("get_price", part_id="P-1042B"),
        "В Казани 3 шт., цена 2630 руб.",
    ])
    result = run_agent("Есть ли P-1042B в Казани и сколько стоит?", llm, tools.as_dict())
    assert tools.count("get_price") == 1
    assert result.answer == "В Казани 3 шт., цена 2630 руб."


def test_explicit_bulletin_preserves_replacement_direction_and_supplier():
    relations = _explicit_relations(
        "Деталь P-4500B (контроллер) снята с производства. Вместо неё устанавливается деталь P-4500C. "
        "Поставщик детали P-4500C — АвтоматикаСевер.", {"Supplier": ["АвтоматикаСевер"]}
    )
    assert {(r.subject.name, r.type, r.object.name) for r in relations} == {
        ("P-4500B", "REPLACED_BY", "P-4500C"), ("P-4500C", "SUPPLIED_BY", "АвтоматикаСевер"),
    }


def test_explicit_bulletin_does_not_infer_replacement_from_negation():
    assert not _explicit_relations(
        "Деталь P-4500 не снята с производства. Вместо неё устанавливается деталь P-4500B.", {}
    )
    assert not _explicit_relations("Поставщик детали P-4500 — не АвтоматикаСевер.",
                                   {"Supplier": ["АвтоматикаСевер"]})


def test_reverse_model_replacement_is_dropped_when_text_is_explicit():
    response = json.dumps({"entities": [], "relations": [
        {"subject_type": "Part", "subject": "P-4500B", "type": "REPLACED_BY",
         "object_type": "Part", "object": "P-4500"},
    ]})
    result = extract_chunk(Chunk("bulletin#0", "bulletin.md",
                                 "Деталь P-4500 снята с производства. Вместо неё устанавливается деталь P-4500B."),
                           FakeLLM([response]))
    assert [(r.subject.name, r.object.name) for r in result.relations] == [("P-4500", "P-4500B")]
    assert result.dropped == 1


def test_supplier_reminder_requests_a_followup_without_guessing_new_part():
    tools = Tools(search_knowledge={"answer": "P-1042B", "facts": [], "sources": [], "refused": False})
    llm = FakeLLM([
        call("search_knowledge", question="Какую деталь заказать для AR-210 при E-17?"),
        "Поставщик не указан.",
        call("search_knowledge", question="Кто поставляет P-1042B?"),
        "В базе знаний не найдено.",
    ])
    result = run_agent("Деталь какого поставщика нужна при E-17 на AR-210?", llm, tools.as_dict())
    assert tools.count("search_knowledge") == 2 and len(llm.requests) == 4
    assert result.refused


def test_fact_number_does_not_confirm_a_stock_claim():
    steps = [
        ToolStep("search_knowledge", {}, result={"answer": "Описание", "facts": ["5. факт"], "refused": False}),
        ToolStep("check_stock", {}, result={"part_id": "P-1042B", "qty": 3}),
    ]
    assert _answer_errors("Осталось 5.", "Проверь P-1042B", steps)
    assert not _answer_errors("Осталось 3.", "Проверь P-1042B", steps)


def test_limit_notice_survives_answer_correction():
    tools = Tools(search_knowledge={"answer": "Цена 9 руб.", "sources": [], "refused": False})
    llm = FakeLLM([call("search_knowledge", question="Цена детали"), "В базе знаний не найдено."])
    result = run_agent("Проверь данные", llm, tools.as_dict(), max_steps=1)
    assert "Работа прервана: достигнут лимит" in result.answer
    assert result.refused and len(result.steps) == 1
    assert len(llm.requests) == 2


def test_real_fact_numbers_are_preserved_but_do_not_prove_stock():
    search = {"answer": "5. Всего 5 отказов.", "facts": ["5. Всего 5 отказов."], "refused": False}
    steps = [ToolStep("search_knowledge", {}, result=search),
             ToolStep("check_stock", {}, result={"part_id": "P-1042B", "qty": 3})]
    before = json.dumps(search, ensure_ascii=False)
    assert not _answer_errors("Всего 5 отказов.", "Проверь P-1042B", steps)
    assert _answer_errors("Осталось 5.", "Проверь P-1042B", steps)
    assert json.dumps(search, ensure_ascii=False) == before


def test_numbered_answer_also_cannot_supply_an_unrelated_number():
    steps = [ToolStep("search_knowledge", {}, result={
        "answer": "5) Описание детали", "facts": [], "refused": False,
    })]
    assert _answer_errors("Получено 5.", "Проверь данные", steps)


def test_limit_notice_survives_failed_answer_correction():
    tools = Tools(search_knowledge={"answer": "Цена 9 руб.", "sources": [], "refused": False})
    llm = FakeLLM([call("search_knowledge", question="Цена детали"), "Цена 7 руб."])
    result = run_agent("Проверь данные", llm, tools.as_dict(), max_steps=1)
    assert result.refused and "Работа прервана: достигнут лимит" in result.answer
    assert "7 руб" not in result.answer and len(llm.requests) == 2


def test_zero_limit_does_not_call_model_and_still_reports_interruption():
    llm = FakeLLM([])
    result = run_agent("Проверь данные", llm, Tools().as_dict(), max_steps=0)
    assert not result.steps and not llm.requests
    assert "Работа прервана: достигнут лимит" in result.answer


def test_order_cannot_use_a_model_only_discovered_by_search():
    tools = Tools()
    llm = FakeLLM([
        call("search_knowledge", question="К какой модели подходит P-1042B?"),
        call("create_service_order", **ORDER),
        "Укажите модель оборудования.",
    ])
    result = run_agent("Оформи заказ: 2 штуки P-1042B, склад Казань.", llm, tools.as_dict(), confirm=True)
    assert tools.count("create_service_order") == 0
    assert result.steps[1].error


def test_order_model_must_be_a_single_model_code():
    tools = Tools()
    llm = FakeLLM([
        call("create_service_order", **{**ORDER, "model": "оборудование"}),
        "Укажите модель оборудования.",
    ])
    result = run_agent("Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.", llm, tools.as_dict(), confirm=True)
    assert tools.count("create_service_order") == 0
    assert result.steps[0].error


def test_prices_cannot_be_swapped_between_parts():
    steps = [
        ToolStep("get_price", {}, result={"part_id": "P-1042B", "price_rub": 2630}),
        ToolStep("get_price", {}, result={"part_id": "P-2054C", "price_rub": 5000}),
    ]
    assert _answer_errors("P-1042B: цена 5000 руб. P-2054C: цена 2630 руб.", "Проверь цены", steps)
    assert not _answer_errors("P-1042B: цена 2630 руб. P-2054C: цена 5000 руб.", "Проверь цены", steps)


def test_quantities_cannot_be_swapped_between_parts():
    steps = [
        ToolStep("check_stock", {}, result={"part_id": "P-1042B", "qty": 3}),
        ToolStep("check_stock", {}, result={"part_id": "P-2054C", "qty": 8}),
    ]
    assert _answer_errors("P-1042B: остаток 8. P-2054C: остаток 3.", "Проверь остатки", steps)
    assert not _answer_errors("P-1042B: остаток 3. P-2054C: остаток 8.", "Проверь остатки", steps)
