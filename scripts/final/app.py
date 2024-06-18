# import streamlit as st
# import torch
# import json
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel, LoraConfig

# # 모델 및 토크나이저 설정
# base_model_name = "meta-llama/Llama-2-13b-chat-hf"
# lora_model_path = "/data/jaesunghwang/sum_termproject/models/finetuned_model4"
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# # 모델 및 토크나이저 불러오기
# st.write("Loading model and tokenizer...")

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=quantization_config,  # 8-bit 양자화 활성화
#     cache_dir="/data/huggingface_models/"
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_name,
#     use_fast=True,
#     cache_dir="/data/huggingface_models/"
# )

# # 패딩 토큰 설정
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # LoRA 모델 설정 로드
# lora_config = LoraConfig.from_pretrained(lora_model_path)

# # LoRA 모델 결합
# model = PeftModel.from_pretrained(base_model, lora_model_path)
# model.eval()

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# def prepare_inputs(texts: list):
#     inputs = []
#     for text in texts:
#         input_text = f"Summarize this document in 1 sentence.\ndocument: {text}\nsummary: "
#         inputs.append(input_text)
#     return inputs

# def process_batch(batch, tokenizer, model):
#     model_inputs = tokenizer(batch, max_length=300, return_tensors="pt", padding=True, truncation=True).to(device)
#     generate_kwargs = {
#         "do_sample": False,
#         "max_new_tokens": 64,
#     }
#     with torch.no_grad():  # No gradient calculation to save memory
#         gen_out = model.generate(**model_inputs, **generate_kwargs)
#     gen_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)  # 디코딩을 호출하여 토큰 ID 목록을 문자열 목록으로 변환
#     return gen_text  # 토큰 ID의 문자열 목록

# def generate_summary(text):
#     instructions = prepare_inputs([text])
#     summaries = process_batch(instructions, tokenizer, model)
#     return summaries[0] if summaries else ""

# # Streamlit 애플리케이션 제목
# st.title("Summarization with LoRA Fine-Tuned Model")

# # 사용자 입력을 위한 텍스트 박스
# user_input = st.text_area("Enter text to summarize:")

# # 분석 버튼
# if st.button("Summarize"):
#     if user_input:
#         summary = generate_summary(user_input)
#         # 결과 출력
#         st.write("### Summary")
#         st.write(summary)
#     else:
#         st.write("Please enter text to summarize.")




import streamlit as st
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig

# 모델 및 토크나이저 설정
base_model_name = "meta-llama/Llama-2-13b-chat-hf"
lora_model_path = "/data/jaesunghwang/sum_termproject/models/finetuned_model6"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 모델 및 토크나이저 불러오기
st.write("Loading model and tokenizer...")

# 8-bit 양자화를 적용하여 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    cache_dir="/data/huggingface_models/"
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_fast=True,
    cache_dir="/data/huggingface_models/"
)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA 모델 설정 로드
lora_config = LoraConfig.from_pretrained(lora_model_path)

# LoRA 모델 결합
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def prepare_inputs(texts: list):
    inputs = []
    for text in texts:
        input_text = f"Summarize this document in 1 sentence.\ndocument: {text}\nsummary: "
        inputs.append(input_text)
    return inputs

def process_batch(batch, tokenizer, model):
    model_inputs = tokenizer(batch, max_length=300, return_tensors="pt", padding=True, truncation=True).to(device)
    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": 64,
    }
    with torch.no_grad():  # No gradient calculation to save memory
        gen_out = model.generate(**model_inputs, **generate_kwargs)
    gen_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)  # 디코딩을 호출하여 토큰 ID 목록을 문자열 목록으로 변환
    return gen_text  # 토큰 ID의 문자열 목록

def post_process_output(instructions, outputs):
    post_processed_output = [output.replace(inst, "").replace("$}}%>", "").replace("$}}%-", "").replace("$}}%", "").replace("#", "").strip() for inst, output in zip(instructions, outputs)]
    return post_processed_output

def generate_summary(text):
    instructions = prepare_inputs([text])
    summaries = process_batch(instructions, tokenizer, model)
    post_processed_summaries = post_process_output(instructions, summaries)
    return post_processed_summaries[0] if post_processed_summaries else ""

# Streamlit 애플리케이션 제목
st.title("Sentence summarization with Llama2-13B-chat LoRA Fine-Tuned model!!")

# 사용자 입력을 위한 텍스트 박스
user_input = st.text_area("Enter Document to summarize:")

# 분석 버튼
if st.button("Summarize"):
    if user_input:
        summary = generate_summary(user_input)
        # 결과 출력
        st.write("### Summary")
        st.write(summary)
    else:
        st.write("Please enter text to summarize.")

