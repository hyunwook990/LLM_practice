from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import chromadb
import json

# 임베딩 모델 설정
embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
client = chromadb.PersistentClient("./chromadb")
collection = client.get_or_create_collection("RAG_doc")

with open ("characters/nara.json", "r", encoding="utf-8") as f:
    RAG_doc = json.load(f)

RAG_vector = embedding_model.encode(RAG_doc).tolist()
collection_exist = collection.get(ids=["doc_0"])

if collection_exist["documents"][0] != RAG_doc[0]:
    collection.add(
        documents=RAG_doc,
        embeddings=RAG_vector,
        ids = [f"doc_{i}" for i in range(len(RAG_doc))]
    )

# 대화 기록 저장
class InMemoryHistory (BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)
    
    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)
    
store = {}

# session_id로 대화 내역 구분
def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


question = input("나:")
query_vector = embedding_model.encode(question).tolist()

search_vectorDB = collection.query(query_embeddings=[query_vector], n_results=5)

retrieved_contexts = search_vectorDB['documents'][0]
context_str = "\n".join(retrieved_contexts)


# prompt 작성 예시
prompt = ChatPromptTemplate.from_messages([
    ('system', "당신은 애니매이션 '나루토'의 등장인물 '시카마루'야. '귀찮다'라는 말이 입버릇이니까 적절히 섞어서 사용해줘 '귀찮지만 어쩔수없지'나 '귀찮게도' 처럼 사용하면 돼."), 
    # seesion_id = 'history'
    MessagesPlaceholder(variable_name='history'),
    ('human', '다음은 당신에 관한 정보입니다. {background} 여기까지의 정보를 토대로 사용자의 질문인 {query}에 잘 대답해주세요.')
])

model = ChatOllama(model="EEVE-Korean-10.8B", temperature=.7)
parser = StrOutputParser()

# prompt, model을 chain으로 묶음
chain = prompt | model | parser

# chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_by_session_id,
    input_messages_key='query',
    history_messages_key='history'
)

response = chain_with_history.invoke(
    {'background': context_str, 'query': question},
    config={'configurable': {'session_id': 'history'}}
)

print(response)