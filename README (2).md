# arxiv-rag-assistant

A RAG system I built for querying AI/ML research papers. Drop in any ArXiv PDF and ask questions in plain English — it finds the right chunks, reranks them, and gives you a grounded answer with page citations.

---

## Why I built this

I got tired of Ctrl+F-ing through 30-page papers trying to find one specific number or methodology detail. I also wanted to go past the basic tutorial RAG pattern (embed → retrieve → generate) and actually implement the techniques that show up in production systems — hybrid search, reranking, evaluation.

Tested it on the Attention Is All You Need paper. Asked it about BLEU scores, optimizer choices, attention head counts — all correct, all cited. Asked it about GPT-3 comparisons (which don't exist in that paper) — it correctly said it couldn't find it. That's when I knew the faithfulness guard was actually working.

---

## Demo

![demo](assets/demo.gif)

---

## What makes this different from a basic RAG tutorial

Most RAG projects online do: embed → retrieve top-k → generate. That's it. Works okay on simple documents, falls apart on dense technical text.

This one adds three things that actually matter:

**Hybrid Search** — BM25 for keyword matching + dense embeddings for semantic similarity, fused with Reciprocal Rank Fusion. The reason you need both: dense retrieval misses exact terms like "BLEU score" or "WMT 2014"; BM25 misses paraphrased or conceptual queries. Combining them covers both failure modes.

**Cross-encoder Reranking** — after retrieval, a `cross-encoder/ms-marco-MiniLM` model re-scores each candidate chunk against the query. Bi-encoders are fast but approximate (they encode query and document separately). Cross-encoders see both together and are much more accurate — worth the extra latency on a shortlist of 10-15 chunks.

**RAGAS-style Evaluation** — faithfulness, answer relevancy, and context recall scored automatically using LLM-as-judge. Most RAG projects have no way to measure if they're actually working. This one does.

You can also switch between fixed and semantic chunking in the UI and see how retrieval quality changes — semantic chunking respects sentence boundaries and works noticeably better on academic text.

---

## Tech stack

| Component | Tool |
|-----------|------|
| LLM | LLaMA 3.3 70B via Groq (free) |
| Embeddings | `all-MiniLM-L6-v2` (local, no API cost) |
| Vector store | ChromaDB (persistent) |
| Sparse retrieval | BM25 via `rank-bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Score fusion | Reciprocal Rank Fusion |
| Evaluation | LLM-as-judge (RAGAS-style) |
| UI | Streamlit |

Everything except the LLM runs locally — no GPU needed, no per-token cost.

---

## Setup

**1. Clone the repo**
```bash
git clone [https://github.com/pavithraathma2002-oss/arxiv-rag-assistant.git]
cd arxiv-rag-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get a free Groq API key**

Go to https://console.groq.com/keys, create a key — it's free, no billing required.

```bash
cp .env.example .env
# paste your key into .env as GROQ_API_KEY=...
```

**4. Run**
```bash
streamlit run app.py
```

---

## How to use it

1. Upload any AI/ML paper PDF in the sidebar (ArXiv papers work great)
2. Pick chunking strategy — Semantic recommended for academic papers
3. Set chunk size to ~800, overlap ~100 for best results
4. Click **Ingest Paper**
5. Ask questions in the Chat tab — answers come with page citations
6. Use **Retrieval Inspector** to see exactly which chunks were retrieved and how they scored
7. Run **RAGAS Evaluation** to get faithfulness and relevancy scores

---

## Project structure

```
arxiv-rag-assistant/
├── app.py                  ← Streamlit UI (3 tabs: chat, inspector, eval)
├── rag/
│   ├── ingest.py           ← PDF extraction + chunking + embedding + ChromaDB
│   ├── retriever.py        ← BM25 + dense + RRF fusion + cross-encoder reranking
│   ├── chain.py            ← LangChain + Groq (LLaMA 3.3 70B) answer generation
│   └── evaluate.py         ← RAGAS-style evaluation (faithfulness, relevancy, recall)
├── chroma_db/              ← persistent vector store (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## What I learned

- Why hybrid search outperforms pure dense retrieval on technical text
- How cross-encoder reranking works and when the latency tradeoff is worth it
- Reciprocal Rank Fusion as a simple, parameter-free way to merge ranked lists
- LLM-as-judge evaluation and why ground-truth metrics are hard to get in RAG
- How chunking strategy meaningfully affects retrieval quality on academic PDFs

---

## Possible improvements

- [ ] Multi-paper support — query across a whole library of papers
- [ ] Streaming responses in the chat UI
- [ ] Export evaluation results to CSV
- [ ] Citation highlighting in the original PDF viewer
- [ ] FAISS as an alternative to ChromaDB for larger corpora
