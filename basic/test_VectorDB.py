from sentence_transformers import SentenceTransformer
import chromadb

# collection list 확인
# client = chromadb.PersistentClient(path="./chromadb")
# print(client.list_collections())

embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

client = chromadb.Client()
collection = client.get_or_create_collection(name="character_info")

documents = [
    "나루토는 라멘을 가장 좋아한다.",
    "사스케는 형을 죽이고 싶어한다.",
    "사쿠라는 사스케를 좋아한다."
]

embeddings = embedding_model.encode(documents)
collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print("문서 추가 완료")

query = "부정적인 감정을 가진 캐릭터는 누구인가?"

query_embedding = embedding_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print(f"검색 결과:")
for doc in results["documents"][0]:
    print("-",doc)