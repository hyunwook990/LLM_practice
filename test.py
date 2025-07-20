import ollama
from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentType, initialize_agent, load_tools

# ollama 기본 세팅
# chat_history = []

# while True:
#     user_input = input("당신: ")

#     if user_input == "종료":
#         break

#     chat_history.append({"role": "user", "content": user_input})

#     response = ollama.chat(
#         model="EEVE-Korean-10.8B",
#         messages=chat_history
#     )

#     print("EEVE: ", response['message']['content'])

#     chat_history.append({"role": "assistant", "content": response['message']['content']})

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

# 사용법 예시
# history_test = get_by_session_id('test')
# history_test.add_messages(['hello', 'good mornig', 'how are you'])
# history_test.add_messages(['I am fine', 'Thank you'])

# prompt 작성 예시
prompt = ChatPromptTemplate.from_messages([
    ('system', '너는 {skill}을 잘하는 AI 어시스턴트야.'), 
    # seesion_id = 'history'
    MessagesPlaceholder(variable_name='history'),
    ('human', '{query}')
])

model = ChatOllama(model="EEVE-Korean-10.8B", temperature=.7)

# prompt, model을 chain으로 묶음
chain = prompt | model

# chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_by_session_id,
    input_messages_key='query',
    history_messages_key='history'
)

response = chain_with_history.invoke(
    {'skill': '대화', 'query': '토끼는 농장에서 나무를 세 그루 키우고 있습니다.'},
    config={'configurable': {'session_id': 'rabbit'}}
)

print(response)

# Agent
tools = load_tools(['wikipedia', 'llm-math'], llm = model)

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_error=True,
    verbose=True
)

print(agent.invoke('애니매이션 나루토의 등장인물 시카마루에 대해 설명해줘'))