import requests
from sentence_transformers import SentenceTransformer
import chromadb

# embedding model로 문서 임베딩
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# chromadb에 문서 저장
# client = chromadb.Client()    # 휘발성 인메모리용
client = chromadb.PersistentClient(path="./chromadb") # 로컬 디스크 저장용 경로 지정 가능

collection = client.get_or_create_collection("character_facts")

documents = [
    "나루토는 라멘을 가장 좋아한다.",
    "사스케는 복수를 위해 마을을 떠났다.",
    "사쿠라는 사스케를 좋아한다."
]

embeddings = embedding_model.encode(documents)

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 사용자 질문 임베딩
user_question = input("사용자: ")

query_vector = embedding_model.encode(user_question).tolist()

# chromadb에서 유사 문서 검색
search_result = collection.query(query_embeddings=[query_vector], n_results=2)

retrieved_contexts = search_result["documents"][0]
context_str = "\n".join(retrieved_contexts)


# 문서 + 질문을 함께 프롬프트로 만들어 ollama에 전송
prompt = f""" 다음은 애니메이션 캐릭터에 대한 설정입니다:
{context_str}

사용자 질문: {user_question}
캐릭터의 성격과 설정에 어긋나지 않도록 자연스럽게 답해주세요.
"""

# ollama 응답 출력
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

print("응답: ")
print(response.json()["response"])

