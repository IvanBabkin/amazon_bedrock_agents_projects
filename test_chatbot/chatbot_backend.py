from langchain.memory import ConversationSummaryBufferMemory
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
        max_tokens=2000)
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

def extract_reasoning_and_response(content):
    """Extract reasoning and main response from the content structure"""
    main_response = ""
    reasoning_content = ""
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                # Check for regular text content
                if item.get('type') == 'text' and 'text' in item:
                    main_response = item['text']
                # Check for reasoning content
                elif item.get('type') == 'reasoning_content' and 'reasoning_content' in item:
                    reasoning_data = item['reasoning_content']
                    if isinstance(reasoning_data, dict) and 'text' in reasoning_data:
                        reasoning_content = reasoning_data['text']
    elif isinstance(content, str):
        # If content is just a string, use it as main response
        main_response = content
    
    return main_response, reasoning_content

def demo_conversation(input_text, memory):
    """Handle conversation with proper system prompt integration and reasoning extraction"""
    
    # Input validation to prevent empty messages
    if not input_text or not input_text.strip():
        return {"main_response": "I need some actual text to respond to. Try asking me something!", "reasoning": None}
    
    # Additional validation for very short inputs that might cause issues
    if len(input_text.strip()) < 2:
        return {"main_response": "Could you be a bit more specific? Single characters don't give me much to work with.", "reasoning": None}
    
    try:
        # Get the LLM instance
        llm = demo_chatbot()
        
        # Get chat history from memory
        chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        
        # Create messages for the conversation
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(chat_history)
        messages.append(HumanMessage(content=input_text.strip()))
        
        # Get response from the model
        response = llm.invoke(messages)
        
        # Extract main response and reasoning content
        main_response, reasoning_content = extract_reasoning_and_response(response.content)
        
        # If no main response was found in content structure, fallback to string conversion
        if not main_response:
            main_response = str(response.content) if response.content else "No response generated."
        
        # Add the conversation to memory (only add the main response, not reasoning)
        memory.chat_memory.add_user_message(input_text.strip())
        memory.chat_memory.add_ai_message(main_response)
        
        return {
            "main_response": main_response,
            "reasoning": reasoning_content if reasoning_content else None
        }
        
    except Exception as e:
        print(f"Error in conversation: {e}")
        return {"main_response": "Sorry, I encountered an error processing your message. Please try rephrasing.", "reasoning": None}
