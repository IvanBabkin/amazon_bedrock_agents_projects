from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# Read system prompt once at module level
def _load_system_prompt():
    with open('test_chatbot/system_prompt.txt', 'r', encoding='utf-8') as f:
        return f.read().strip()

SYSTEM_PROMPT = _load_system_prompt()

def demo_chatbot():
    """Create and return the LLM instance"""
    demo_llm = ChatBedrockConverse(
        credentials_profile_name='default',
        model="us.deepseek.r1-v1:0",
        temperature=0.1,
        max_tokens=1000)
    return demo_llm

def demo_memory():
    """Create conversation memory with the LLM"""
    llm_data = demo_chatbot()
    memory = ConversationSummaryBufferMemory(
        llm=llm_data,
        max_token_limit=2000,
        return_messages=True,  # Important for chat models
        memory_key="chat_history"
    )
    return memory

def demo_conversation(input_text, memory):
    """Handle conversation with proper system prompt integration"""
    
    # Input validation to prevent empty messages
    if not input_text or not input_text.strip():
        return "I need some actual text to respond to. Try asking me something!"
    
    # Additional validation for very short inputs that might cause issues
    if len(input_text.strip()) < 2:
        return "Could you be a bit more specific? Single characters don't give me much to work with."
    
    # Create the chat prompt template with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create conversation chain with the prompt template
    llm = demo_chatbot()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    
    # Invoke with just the user input
    try:
        chat_reply = conversation.invoke({"input": input_text.strip()})
        return chat_reply['response']
    except Exception as e:
        print(f"Error in conversation: {e}")
        return "Sorry, I encountered an error processing your message. Please try rephrasing."
