"""
evaluate.py — RAGAS-style evaluation of RAG pipeline.
Measures: Faithfulness, Answer Relevancy, Context Recall.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from rag.retriever import hybrid_search
from rag.chain import get_answer


def score_faithfulness(answer: str, context: str, llm) -> float:
    """
    Faithfulness: Are all claims in the answer supported by the context?
    Uses LLM-as-judge approach.
    """
    prompt = f"""Given the following context and answer, score the faithfulness of the answer.
Faithfulness means: every claim in the answer is directly supported by the context.

Context:
{context}

Answer:
{answer}

Score from 0.0 to 1.0 where:
- 1.0 = every claim is fully supported by context
- 0.5 = some claims are supported, some are not
- 0.0 = answer contradicts or ignores the context

Respond with ONLY a number between 0.0 and 1.0."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return float(response.content.strip())
    except:
        return 0.5


def score_answer_relevancy(question: str, answer: str, llm) -> float:
    """
    Answer Relevancy: Does the answer actually address the question?
    """
    prompt = f"""Score how well the answer addresses the question.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0 where:
- 1.0 = answer directly and completely addresses the question
- 0.5 = answer partially addresses the question
- 0.0 = answer is irrelevant to the question

Respond with ONLY a number between 0.0 and 1.0."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return float(response.content.strip())
    except:
        return 0.5


def score_context_recall(ground_truth: str, context: str, llm) -> float:
    """
    Context Recall: Does the retrieved context contain info needed to answer?
    Only computed when ground truth is provided.
    """
    if not ground_truth or not ground_truth.strip():
        return -1  # skip

    prompt = f"""Given the ground truth answer and the retrieved context, score context recall.
Context recall measures whether the retrieved context contains the information needed to produce the ground truth answer.

Ground truth: {ground_truth}
Retrieved context: {context}

Score from 0.0 to 1.0 where:
- 1.0 = context contains all information needed
- 0.5 = context contains some relevant information
- 0.0 = context is missing key information

Respond with ONLY a number between 0.0 and 1.0."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return float(response.content.strip())
    except:
        return 0.5


def evaluate_rag(
    eval_data: list[dict],
    collection_name: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_reranker: bool = True,
) -> dict:
    """
    Run full RAGAS evaluation over a list of {question, ground_truth} pairs.

    Returns aggregated + per-question scores.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.0,
        max_tokens=1024,
    )

    faithfulness_scores = []
    relevancy_scores = []
    recall_scores = []
    details = []

    for item in eval_data:
        question = item.get("question", "").strip()
        ground_truth = item.get("ground_truth", "").strip()

        if not question:
            continue

        # Retrieve
        chunks = hybrid_search(
            question,
            collection_name,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

        # Build context
        context = "\n\n".join([c["text"] for c in chunks])

        # Generate answer
        answer = get_answer(question, chunks)

        # Score
        f_score = score_faithfulness(answer, context, llm)
        r_score = score_answer_relevancy(question, answer, llm)
        rc_score = score_context_recall(ground_truth, context, llm)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        if rc_score >= 0:
            recall_scores.append(rc_score)

        details.append({
            "question": question,
            "answer": answer,
            "faithfulness": f_score,
            "relevancy": r_score,
            "context_recall": rc_score if rc_score >= 0 else None,
        })

    return {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
        "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
        "context_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        "details": details,
    }
