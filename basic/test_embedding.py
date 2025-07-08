from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

sentence = [
    "나루토는 사스케를 라이벌로 생각한다.",
    "사스케는 일족의 복수를 꿈꾼다.",
    "사쿠라는 사스케를 좋아한다."
]

embeddings = embedding_model.encode(sentence)

query = "사스케를 라이벌로 생각하는 사람은?"

query_embedding = embedding_model.encode(query)

score = util.cos_sim(query_embedding, embeddings)

for idx, sco in enumerate(score[0]):
    print(f"문장 {idx+1} 유사도 점수: {sco:.4f}")