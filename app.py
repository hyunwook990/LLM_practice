import streamlit as st
import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 캐릭터별 아이콘
character_avatars = {
    "별주부전 - 토끼": "🐰",
    "별주부전 - 자라": "🐢",
    "별주부전 - 용왕": "🐲",
    "흥부와 놀부전 - 흥부": "😊",
    "흥부와 놀부전 - 놀부": "😠",
    "해와 달이 된 오누이 - 오빠": "👦",
    "해와 달이 된 오누이 - 동생": "👧",
    "해와 달이 된 오누이 - 호랑이": "🐯",
    "하이큐 - 히나타 쇼요":"🐦‍⬛",
    "블리치 - 이치마루 긴":"3️⃣",
    "나루토 - 우치하 이타치":"🥷",
    "원피스 - 롤로노아 조로":"🔪🔪🔪",
    "📝 직접 입력": "✍️"
}

# JSON 파일에서 캐릭터 프롬프트 불러오기
with open("prompts.json", "r", encoding="utf-8") as f:
    characters = json.load(f)

# Streamlit UI
st.set_page_config("🧚 전래동화 캐릭터와 대화하기", layout="centered")
st.title("🧚 전래동화 속 캐릭터와 대화하기")

# 캐릭터 선택
options = list(characters.keys()) + ["📝 직접 입력"]
selected = st.selectbox("📖 대화하고 싶은 캐릭터를 선택하세요", options)

# 사용자 직접 입력
if selected == "📝 직접 입력":
    custom_prompt = st.text_area("✍️ 캐릭터의 성격이나 상황을 자유롭게 입력해보세요", height=150)
    if not custom_prompt.strip():
        st.warning("⚠️ 캐릭터 설명을 입력해야 대화가 가능합니다.")
        st.stop()
    final_prompt = (
        custom_prompt.strip() + "\n\n"
        "너는 너 자신이 이 설정에 해당하는 인물이라고 철저히 믿고 행동해야 해. "
        "AI나 프로그램이라는 말은 절대 해서는 안 되며, 너는 그 캐릭터를 연기하는 게 아니라 그 인물이 되어야 해. "
        "너무 길게 답변하지 마. 'Human:'이나 'Assistant:' 같은 문답 형식은 절대 사용하지 마. "
        "사용자의 말에만 반응하고, 스스로 질문하거나 설명하지 마."
    )
else:
    final_prompt = characters[selected]

# 모델 초기화
model = ChatOllama(model="EEVE-Korean-10.8B")
parser = StrOutputParser()

# 상태 초기화
if "chat_history" not in st.session_state or st.session_state.get("selected_character") != selected:
    st.session_state.chat_history = [
        {"role": "system", "content": final_prompt}
    ]
    st.session_state.selected_character = selected

# 대화 출력
for msg in st.session_state.chat_history[1:]:
    if msg["role"] == "user":
        st.chat_message("user", avatar="🙋").markdown(msg["content"])
    else:
        st.chat_message("assistant", avatar=character_avatars.get(selected, "🧙")).markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("무엇을 물어볼까요?")
if user_input:
    st.chat_message("user", avatar="🙋").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    prompt = ChatPromptTemplate.from_messages(st.session_state.chat_history)
    response = (prompt | model | parser).invoke({})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.chat_message("assistant", avatar=character_avatars.get(selected, "🧙")).markdown(response)
