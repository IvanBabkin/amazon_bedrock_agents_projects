# Source:Below code is provided by Streamlit and AWS 

import streamlit as st 
import  chatbot_backend as demo

st.title("ChatGloom2000")

if 'memory' not in st.session_state: 
    st.session_state.memory = demo.demo_memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history: 
    with st.chat_message(message["role"], avatar=message.get("avatar")): 
        st.markdown(message["text"]) 

input_text = st.chat_input("Chat with ChatGloom2000 here")
if input_text and input_text.strip(): 
    
    with st.chat_message("user"): 
        st.markdown(input_text) 
    
    st.session_state.chat_history.append({
        "role":"user", 
        "text":input_text}) 

    chat_response = demo.demo_conversation(
        input_text=input_text,
        memory=st.session_state.memory)
    
    # Display reasoning FIRST if available
    if chat_response["reasoning"]:
        with st.chat_message("assistant", avatar="test_chatbot/DALL_E_ChatGloom2000.png"):
            st.markdown("🧠 **Model's Internal Reasoning:**")
            st.markdown(chat_response["reasoning"])
        
        st.session_state.chat_history.append({
            "role":"assistant", 
            "text": f"🧠 **Reasoning:** {chat_response['reasoning']}", 
            "avatar":"test_chatbot/DALL_E_ChatGloom2000.png"})
    
    # Display main response AFTER reasoning
    with st.chat_message("assistant", avatar="test_chatbot/DALL_E_ChatGloom2000.png"): 
        st.markdown(chat_response["main_response"]) 
    
    st.session_state.chat_history.append({
        "role":"assistant", 
        "text":chat_response["main_response"], 
        "avatar":"test_chatbot/DALL_E_ChatGloom2000.png"}) 