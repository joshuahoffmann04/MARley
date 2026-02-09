from __future__ import annotations

from pathlib import Path

class PromptBuilder:
    def __init__(self, prompts_dir: Path | None = None) -> None:
        self.prompts_dir = prompts_dir
        self.default_system_prompt = (
            "Du bist MARley, ein Studienberatungs-Assistent für die Studienordnung. "
            "Antworte immer auf Deutsch. "
            "Nutze ausschließlich den bereitgestellten Kontext. "
            "Wenn der Kontext nicht ausreicht oder uneindeutig ist, dann abstain. "
            "Antworte ausschließlich als JSON gemäß dem vorgegebenen Schema."
        )

    def _load_prompt(self, filename: str, default: str) -> str:
        if self.prompts_dir:
             path = self.prompts_dir / filename
             if path.exists():
                 return path.read_text(encoding="utf-8").strip()
        return default

    def build_system_prompt(self) -> str:
        return self._load_prompt("system_prompt.txt", self.default_system_prompt)

    def build_user_prompt(self, question: str, context_block: str) -> str:
        base_user_prompt = (
            "Frage:\n"
            "{question}\n\n"
            "Kontext:\n"
            "{context_block}\n\n"
            "Regeln:\n"
            "1) Keine Halluzinationen.\n"
            "2) Wenn unklar, setze should_abstain=true und answer kurz.\n"
            "3) Antwort knapp und präzise auf Deutsch."
        )
        template = self._load_prompt("user_prompt_template.txt", base_user_prompt)
        # Safe format in case template has other braces?
        # Assuming simple f-string style replacement manually or using format
        return template.format(question=question.strip(), context_block=context_block)
