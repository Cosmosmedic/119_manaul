import streamlit as st
from app.rag_pipeline import ManualRAG
from app.loader import load_algorithms, load_algorithm_image
from app.config import GPT_MODEL, OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)


# GPT 답변 생성
def generate_answer(query, context):
    prompt = f"""
다음 내용은 '119 구급대원 현장응급처치 표준지침' 일부입니다.

[검색된 지침 일부]
{context}

[사용자 질문]
{query}

위 지침을 기반으로 정확하고 전문적으로 답변하세요.
"""

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "너는 응급의학과 전문의이며 119 구급대원 지침을 가장 잘 이해하고 있는 전문가다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return completion.choices[0].message["content"]


# Streamlit 화면 구성
def render_app():
    st.title("🚑 119 응급처치 RAG Assistant")
    st.markdown("119 구급대원 응급처치 표준지침 기반 Q/A AI")

    query = st.text_input("질문을 입력하세요 (예: 성인 심정지 알고리즘 알려줘)")

    if not query:
        return

    # RAG 초기화 (속도 향상 위해 세션에 저장)
    if "rag" not in st.session_state:
        rag = ManualRAG()
        rag.build_index()
        st.session_state["rag"] = rag

    rag = st.session_state["rag"]

    # 🔍 PDF 텍스트 검색
    with st.spinner("지침에서 관련 내용을 검색하고 있습니다..."):
        results = rag.search(query, top_k=3)

    context_text = "\n\n".join([r["text"] for r in results])

    # 🧠 GPT 답변 생성
    answer = generate_answer(query, context_text)

    st.markdown("## 🩺 AI 답변")
    st.write(answer)

    # 📊 알고리즘 이미지 매칭
    algorithms = load_algorithms()
    matched = None

    for item in algorithms:
        if item["title"].replace(" ", "") in query.replace(" ", ""):
            matched = item
            break

    if matched:
        st.markdown("---")
        st.subheader(f"📊 알고리즘 이미지: {matched['title']} (p.{matched['page']})")

        img = load_algorithm_image(matched["page"])
        if img:
            st.image(img)
        else:
            st.warning("이미지를 아직 업로드하지 않았습니다.")

