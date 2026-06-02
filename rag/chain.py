"""
chain.py — LangChain RAG chain using Groq (free, fast).
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

PROMPT_TEMPLATE = """You are an expert research assistant helping users understand AI/ML papers.

Answer the question based ONLY on the following context from the paper.
If the answer is not in the context, say "I couldn't find this in the paper."
Be precise, cite page numbers when possible, and avoid hallucinating.

Context:
{context}

Question: {question}

Answer:"""


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=1024,
    )


def get_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found in the paper."

    context_parts = []
    for chunk in chunks:
        page = chunk.get("page", "?")
        context_parts.append(f"[Page {page}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"