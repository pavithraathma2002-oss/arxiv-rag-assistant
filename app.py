import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag.ingest import ingest_pdf
from rag.retriever import hybrid_search
from rag.chain import get_answer
from rag.evaluate import evaluate_rag

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e0e0e0;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #00ff88;
}

.metric-card {
    background: #111118;
    border: 1px solid #00ff8833;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    color: #00ff88;
    font-weight: 600;
}

.metric-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.answer-box {
    background: #111118;
    border-left: 3px solid #00ff88;
    border-radius: 4px;
    padding: 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.7;
}

.source-chip {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #00ff8855;
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #00ff88;
    margin: 4px 4px 4px 0;
}

.chunk-box {
    background: #0d0d1a;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 14px;
    font-size: 0.82rem;
    color: #aaa;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.6;
}

.stButton > button {
    background: #00ff88;
    color: #0a0a0f;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #00cc6a;
    transform: translateY(-1px);
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #111118;
    border: 1px solid #333;
    color: #e0e0e0;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 4px;
}

.stSelectbox > div > div {
    background: #111118;
    border: 1px solid #333;
    color: #e0e0e0;
}

.sidebar .sidebar-content {
    background: #080810;
}

.tag {
    display: inline-block;
    background: #00ff8815;
    border: 1px solid #00ff8840;
    color: #00ff88;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔬 ArXiv RAG")
    st.markdown("<span class='tag'>hybrid search</span><span class='tag'>reranking</span><span class='tag'>RAGAS</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📄 Upload Paper")
    uploaded_file = st.file_uploader("Drop an AI/ML paper (PDF)", type=["pdf"])

    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        ["Fixed (512 tokens)", "Semantic (sentence-aware)"],
        help="Fixed splits by token count. Semantic splits on sentence boundaries."
    )

    chunk_size = st.slider("Chunk size (tokens)", 256, 1024, 512, 64)
    chunk_overlap = st.slider("Overlap (tokens)", 0, 200, 50, 10)
    top_k = st.slider("Top-K retrieval", 2, 10, 5)

    use_reranker = st.toggle("Cross-encoder reranking", value=True)
    use_hybrid = st.toggle("Hybrid search (BM25 + dense)", value=True)

    if uploaded_file and st.button("⚡ Ingest Paper"):
        with st.spinner("Chunking, embedding, indexing..."):
            strategy = "semantic" if "Semantic" in chunking_strategy else "fixed"
            collection_name = ingest_pdf(
                uploaded_file,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                strategy=strategy,
            )
            st.session_state.ingested = True
            st.session_state.collection_name = collection_name
            st.session_state.messages = []
            st.session_state.eval_results = None
        st.success(f"✅ Ingested: {uploaded_file.name}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password",
                          value=os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("---")
    st.markdown("<div style='color:#444;font-size:0.75rem;font-family:IBM Plex Mono'>built by sidharth-1005</div>", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Retrieval Inspector", "📊 RAGAS Evaluation"])

# ── TAB 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Ask the Paper")

    if not st.session_state.ingested:
        st.info("👈 Upload a PDF in the sidebar to get started.")
    else:
        # Chat history
        for msg in st.session_state.messages:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Ask anything about the paper..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving + generating..."):
                    chunks = hybrid_search(
                        prompt,
                        st.session_state.collection_name,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                    )
                    answer = get_answer(prompt, chunks)

                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                # Sources
                st.markdown("**Sources:**")
                for c in chunks[:3]:
                    page = c.get("page", "?")
                    st.markdown(f"<span class='source-chip'>📄 Page {page}</span>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ── TAB 2: Retrieval Inspector ────────────────────────────────────────────────
with tab2:
    st.markdown("## Retrieval Inspector")
    st.markdown("See exactly what chunks are retrieved and how they're ranked.")

    if not st.session_state.ingested:
        st.info("👈 Upload and ingest a PDF first.")
    else:
        inspect_query = st.text_input("Enter a query to inspect retrieval:")

        col1, col2 = st.columns(2)
        with col1:
            show_bm25 = st.toggle("Show BM25 scores", value=True)
        with col2:
            show_rerank = st.toggle("Show rerank scores", value=True)

        if inspect_query:
            with st.spinner("Running retrieval pipeline..."):
                chunks = hybrid_search(
                    inspect_query,
                    st.session_state.collection_name,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                    return_scores=True,
                )

            st.markdown(f"**{len(chunks)} chunks retrieved**")
            for i, c in enumerate(chunks):
                with st.expander(f"Chunk {i+1} — Page {c.get('page', '?')}"):
                    st.markdown(f"<div class='chunk-box'>{c['text']}</div>", unsafe_allow_html=True)
                    cols = st.columns(3)
                    if show_bm25:
                        cols[0].metric("BM25 Score", f"{c.get('bm25_score', 0):.3f}")
                    cols[1].metric("Dense Score", f"{c.get('dense_score', 0):.3f}")
                    if show_rerank:
                        cols[2].metric("Rerank Score", f"{c.get('rerank_score', 0):.3f}")

# ── TAB 3: RAGAS Evaluation ───────────────────────────────────────────────────
with tab3:
    st.markdown("## RAGAS Evaluation")
    st.markdown("Automatically evaluate RAG quality on faithfulness, answer relevancy, and context recall.")

    if not st.session_state.ingested:
        st.info("👈 Upload and ingest a PDF first.")
    else:
        st.markdown("### Test Questions")
        st.markdown("Add question + ground truth pairs to evaluate:")

        default_questions = [
            {"question": "What is the main contribution of this paper?", "ground_truth": ""},
            {"question": "What datasets were used in the experiments?", "ground_truth": ""},
            {"question": "What are the limitations of the proposed method?", "ground_truth": ""},
        ]

        eval_data = []
        for i, q in enumerate(default_questions):
            with st.expander(f"Test case {i+1}"):
                question = st.text_input(f"Question {i+1}", value=q["question"], key=f"q_{i}")
                ground_truth = st.text_area(f"Ground truth {i+1} (optional)", key=f"gt_{i}", height=80)
                eval_data.append({"question": question, "ground_truth": ground_truth})

        if st.button("🧪 Run RAGAS Evaluation"):
            with st.spinner("Running evaluation pipeline... this takes ~30 seconds"):
                results = evaluate_rag(
                    eval_data,
                    st.session_state.collection_name,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                )
                st.session_state.eval_results = results

        if st.session_state.eval_results:
            r = st.session_state.eval_results
            st.markdown("### Results")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{r['faithfulness']:.2f}</div>
                    <div class='metric-label'>Faithfulness</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{r['answer_relevancy']:.2f}</div>
                    <div class='metric-label'>Answer Relevancy</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{r['context_recall']:.2f}</div>
                    <div class='metric-label'>Context Recall</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("### Per-question breakdown")
            for item in r.get("details", []):
                with st.expander(item["question"]):
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.markdown(f"**Faithfulness:** `{item['faithfulness']:.2f}` | **Relevancy:** `{item['relevancy']:.2f}`")
