# ui/tab_chat.py
import hashlib
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain

def render_chat_tab(model_name: str, kb_text: str, kb_hash: str):
    st.subheader("💬 Chat with the FPL Agent")

    def _make_chain(api_key: str, kb_text: str):
        llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.2)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """You are an FPL expert with access to a knowledge base of real-time player stats and upcoming fixtures.
Use only this knowledge base to provide data-driven advice. Be concise and specific.
Core duties: Recommend team, transfers, captain and chip strategies. Use injury/rotation info.
Rules: Max 3 per club, respect budgets. EPL data only.
Response: Give concrete picks with prices and reasoning; reference current GW and upcoming runs."""
                ),
                ("system", kb_text),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

    api_key = st.session_state.openai_key
    current_hash = hashlib.sha256(kb_text.encode("utf-8")).hexdigest()

    if ("conversation" not in st.session_state) or (st.session_state.get("kb_hash") != current_hash) or st.button("Rebuild chat with current KB"):
        if api_key:
            st.session_state.conversation = _make_chain(api_key, kb_text)
            st.session_state.kb_hash = current_hash
        else:
            st.info("Enter your OpenAI API key in the sidebar to enable chat.")

    if "conversation" in st.session_state:
        mem_msgs = st.session_state.conversation.memory.chat_memory.messages
        for m in mem_msgs:
            st.chat_message("user" if m.type == "human" else "assistant").write(m.content)

    user_input = st.chat_input("Ask about FPL (e.g., best £6.5m mids, who to captain, wildcard draft)...")
    if user_input:
        st.chat_message("user").write(user_input)
        if not api_key:
            assistant_reply = "Please enter your OpenAI API key in the sidebar to use the chat agent."
        elif "conversation" not in st.session_state:
            assistant_reply = "Chat not initialized. Click 'Rebuild chat with current KB' after entering your API key."
        else:
            with st.spinner("Thinking..."):
                try:
                    assistant_reply = st.session_state.conversation.predict(input=user_input)
                except Exception as e:
                    assistant_reply = f"Error: {e}"
        st.chat_message("assistant").write(assistant_reply)
