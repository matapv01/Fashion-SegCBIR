from typing import List


def rewrite_query(llm, base_query: str, history: List[dict]) -> str:
        if not history:
            return base_query
        history_str = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in history])
        prompt = f"""Based on chat history and the user's latest request, generate a concise fashion query. Return only the query.
                Chat history:
                {history_str}
                - user: {base_query}
                New search query:"""
        return llm.chat(messages=[{"role":"user","content":prompt}]).strip()