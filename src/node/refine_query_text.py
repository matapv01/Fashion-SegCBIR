from typing import List


def refine_query_text(llm, base_query: str, feedback_text: str) -> str:
        prompt = f"The user initial: {base_query}, feedback: {feedback_text}. Generate new concise query."
        return llm.chat(messages=[{"role":"user","content":prompt}]).strip()