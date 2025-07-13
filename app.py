import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import json

# 임베딩 모델 및 ChromaDB 초기화
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
client = chromadb.PersistentClient("./chromadb")
collection = client.get_or_create_collection("character_background_shikamaru")

# 문서 로드 및 초기 등록 (중복 방지)
with open("characters/nara.json", "r", encoding="utf-8") as f:
    nara_character = json.load(f)

embeddings_nara = embedding_model.encode(nara_character).tolist()

if not collection.get(ids=["doc_0"])["documents"]:
    collection.add(
        documents=nara_character,
        embeddings=embeddings_nara,
        ids=[f"doc_{i}" for i in range(len(nara_character))]
    )

# Streamlit UI
st.title("캐릭터와 대화하기")
question = st.text_input("나: ", placeholder="질문을 입력해보세요!")

if st.button("시카마루에게 질문하기") and question:
    # 벡터 검색
    query_vector = embedding_model.encode(question).tolist()
    search_result = collection.query(query_embeddings=[query_vector], n_results=3)
    retrieved_contexts = search_result["documents"][0]
    context_str = "\n".join(retrieved_contexts)

    # 프롬프트 구성
    prompt = f"""다음은 애니메이션 캐릭터에 대한 설정입니다:
{context_str}
가장 큰 특징은 귀찮아하지만 책임감이 강한 성격이야.
대답할 때 "귀찮지만", "귀찮게도"와 같은 말투를 섞어줘.

사용자 질문: {question}
캐릭터의 성격과 설정에 어긋나지 않도록 자연스럽게 답해주세요.
"""

    # Ollama 호출
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "EEVE-Korean-10.8B",
            "prompt": prompt,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            },
            "stream": False
        }
    )

    answer = response.json()["response"]
    st.markdown("### 시카마루:")
    st.write(answer)
