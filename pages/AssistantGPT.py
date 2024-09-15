from openai import OpenAI 

import requests
from langchain.retrievers import WikipediaRetriever


import requests
import streamlit as st
import json

 
def search_wikipedia(inputs):
    keyword = inputs["keyword"]
    retriever = WikipediaRetriever(top_k_results=3, lang="en")
    data_list = retriever.get_relevant_documents(keyword)
    results = "Wikipedia Result\n\n"
    for page_content in data_list:
        results += f"{page_content.page_content}\n\n"
    return results



tools_map ={
    "search_wikipedia": search_wikipedia,

}

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Given a keyword, returns the top 3 relevant Wikipedia page contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword to search on Wikipedia.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
]




@st.cache_resource
def get_assistants(_client):
    return _client.beta.assistants.create(
        name="Research Assistant",
        instructions="You are a personal Research Assistant. You help users do research on topics.",
        model="gpt-4o-mini",
        tools=tools,
    )

def is_valid_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        return response.status_code ==200
    except requests.RequestException:
        return False



def draw_message(message,role):
    with st.chat_message(role):
        st.markdown(message)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message,role):
    draw_message(message,role)
    save_message(message,role)

def paint_history():
    for message in st.session_state["messages"]:
        draw_message(
            message["message"],
            message["role"],
        )

def extract_message_content(message):
    if message.content and isinstance(message.content, list):
        return "\n\n".join([block.text.value for block in message.content if hasattr(block, 'text')])
    return "이용 가능한 컨텐츠가 없습니다."


st.set_page_config(page_title="Research GPT", page_icon="🔍")



if 'messages' not in st.session_state:
    st.session_state['messages'] = []


with st.sidebar:
    st.markdown("""
                코드: pages/AssistantGPT
                https://github.com/LikeRudin/fullstack_gpt
""")
    api_key = st.text_input("OpenAI API 키 입력:", type="password")


st.title("Research GPT")
st.markdown("""
            안녕하세요!.
            Research Assistant GPT입니다.
            연구하고싶은 주제의 키워드를 입력해주세요.""")



if api_key:
    if is_valid_api_key(api_key):
        st.sidebar.success("API 키가 유효합니다.")
        
        client = OpenAI(api_key=api_key)
        
        paint_history()
        user_input = st.chat_input("연구자료가 필요한 keyword를 입력해주세요")
        if user_input:
           
            send_message(role="user", message=user_input)
           
            assistant = get_assistants(_client=client)

            thread = client.beta.threads.create()

            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content= f"Research about {user_input}"
            )

            run = client.beta.threads.runs.create_and_poll(
               thread_id=thread.id,
               assistant_id=assistant.id,)
            send_message(role="ai", message=f"{user_input}에 대한 연구를 진행합니다")
            
            while run.status != 'completed':
                if run.required_action.type == 'submit_tool_outputs':
                    tool_outputs = []
                    for tool in run.required_action.submit_tool_outputs.tool_calls:
                        function = tool.function
                        tool_id =  tool.id
                        tool_name = function.name
                        tool_inputs = json.loads(function.arguments)
                        send_message(role="ai", message=f"'{tool_name}'  tool을 사용합니다. 다음 argument를 사용합니다. {tool_inputs}")
                        if tool_name in tools_map:
                            tool_output = tools_map[tool_name]
                            tool_outputs.append({
                                "tool_call_id": tool_id,
                                "output": tool_output((tool_inputs))
                            })
                        else:
                            tool_outputs.append({
                                "tool_call_id": tool_id,
                                "output": f"Error:  '{tool_name}' Tool 을 찾을 수없습니다."
                            })
                    run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs)
                else:
                    run = client.beta.threads.runs.poll(thread_id=thread.id, run_id=run.id)
  
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                messages_list = list(messages)
                assistant_reply = messages_list[0]
                formatted_reply = extract_message_content(assistant_reply)
                print(assistant_reply)
                send_message(role="ai", message="연구가 완료되었습니다.")
                send_message(role="ai", message=formatted_reply)
            else:
                send_message(role="ai", message=f"'{user_input}'에 관한 연구가 실패했습니다 ") 
    else:
        st.sidebar.error("유효하지 않은 OpenAI API 키입니다. 다시 시도해주세요.")
else:
    st.sidebar.warning("OpenAI API 키를 입력해주세요.")