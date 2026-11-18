"""Диагностический прогон без кеша: метрики, полные ответы и трасса LLM."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from assistant.infra.cli import Context, runner
from assistant.infra.config import Settings
from assistant.infra.erp_client import ErpClient
from assistant.infra.evaluation import evaluate
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["agent", "global", "local", "baseline"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/questions_dev.jsonl"))
    args = parser.parse_args()
    settings = Settings.from_env()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=False)  # Не перезаписываем предыдущий прогон.
    answers = args.out / "answers"
    answers.mkdir()
    started = time.monotonic()
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
        "mode": args.mode, "cache_dir": None, "confirm": False,
        "model": settings.llm_model, "embed_model": settings.embed_model,
        "llm_base_url": settings.llm_base_url, "erp_url": settings.erp_url,
        "questions": str(args.questions), "hashes": {},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    paths = sorted(Path("src/assistant/agent").glob("*.py"))
    paths += sorted(Path("src/assistant/graphrag").glob("*.py"))
    paths += sorted((settings.data_dir / "corpus").glob("*.md"))
    paths += [args.questions, settings.data_dir / "vocabulary.json",
              settings.artifacts_dir / "graph.json", settings.artifacts_dir / "communities.json"]
    for path in paths:
        manifest["hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_relative_to("src"):
            target = args.out / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    health = httpx.get(settings.erp_url + "/health", timeout=5)
    health.raise_for_status()
    tags = httpx.get(settings.llm_base_url.removesuffix("/v1") + "/api/tags", timeout=5)
    tags.raise_for_status()
    manifest["erp_health"] = health.json()
    manifest["ollama_models"] = [{key: model[key] for key in ("name", "digest")}
                                 for model in tags.json()["models"]]
    names = {model["name"] for model in tags.json()["models"]}
    if settings.llm_model not in names:
        raise RuntimeError("Требуемая модель отсутствует в Ollama.")
    if args.mode == "baseline" and not ({settings.embed_model, settings.embed_model + ":latest"} & names):
        raise RuntimeError("Модель эмбеддингов отсутствует в Ollama.")
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    llm = LLMClient(settings.llm_base_url, settings.llm_model, settings.embed_model,
                    api_key=settings.llm_api_key, cache_dir=None)
    original_chat = llm.chat
    active_question = ""

    def traced(messages, **kwargs):
        before = time.monotonic()
        response = original_chat(messages, **kwargs)
        with (args.out / "llm-trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "question_id": active_question, "elapsed_s": time.monotonic() - before,
                "messages": messages, "options": kwargs, "response": response.as_message(),
            }, ensure_ascii=False) + "\n")
        return response

    llm.chat = traced
    ask = runner(args.mode, Context(settings, llm, ErpClient(settings.erp_url)), confirm=False)

    def measured(question):
        nonlocal active_question
        active_question = question["id"]
        before = time.monotonic()
        outcome = ask(question["question"])
        result = asdict(outcome)
        result["answer"] = result.pop("text")
        record = {"id": active_question, "question": question["question"],
                  "elapsed_s": time.monotonic() - before, "result": result}
        (answers / f"{active_question}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"{args.mode} {active_question}: refused={outcome.refused}, "
              f"pending={outcome.needs_confirmation}", flush=True)
        return outcome

    known = {code for path in (settings.data_dir / "corpus").glob("*.md")
             for code in find_codes(path.read_text(encoding="utf-8"))}
    report = evaluate(questions, measured, known, llm.stats)
    report["summary"]["mode"] = args.mode
    (args.out / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.update({"finished_utc": datetime.now(UTC).isoformat(), "elapsed_s": time.monotonic() - started,
                     "llm_stats": llm.stats.as_dict()})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
