import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 侧边栏设置
with st.sidebar:
    st.markdown("## InternLM LLM")
    st.markdown("[GitHub](https://github.com/ljn12yyds/zhangsan_say_law)")
    # 创建一个滑块，用于选择最大长度，范围在0到1024之间，默认值为512
    max_length = st.slider("Max Length", 0, 1024, 512, step=1)
    system_prompt = st.text_input("System Prompt", "你是一个由上海人工智能实验室提供支持开发的法律大模型。现在你是一个经验丰富的法律学专家，名叫张三。我有一些关于法律方面的问题，请你用专业的知识引用相关的法律法规帮我解决")

# 创建一个标题和一个副标题
st.title("💬 InternLM2-Chat-7B 张三普法")
st.caption("🚀 A streamlit chatbot powered by InternLM2 QLora")

# 定义模型路径
model_id = 'ljnyyds/zhangsan_say_law'

# 获取模型和分词器
@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model

tokenizer, model = get_model()

# 初始化消息列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示历史消息
for i, (user_message, assistant_message) in enumerate(st.session_state.messages):
    st.text_area(f"User message {i}", value=user_message, height=75)
    st.text_area(f"Assistant message {i}", value=assistant_message, height=75)

# 获取用户输入
prompt = st.text_input("请输入你的问题：", key="new_message")

# 如果用户输入了内容，则执行以下操作
if prompt:
    # 构建输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    
    # 生成模型响应
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length)
    
    # 解码模型响应
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 将用户输入和模型响应添加到 session_state 中的 messages 列表中
    st.session_state.messages.append((prompt, response))
    
    # 清空输入框
    st.session_state["new_message"] = ""
