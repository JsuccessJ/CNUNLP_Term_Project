import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 모델 및 토크나이저 설정
model_name = "meta-llama/Llama-2-13b-chat-hf"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 모델 및 토크나이저 불러오기
print("Loading model and tokenizer...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,  # 8-bit 양자화 활성화
    cache_dir="/data/huggingface_models/"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    cache_dir="/data/huggingface_models/"
)

# LoRA 모델 로드 및 결합
lora_model_path = "/data/jaesunghwang/sum_termproject/models/finetuned_model6"
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.to(device)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model and tokenizer loaded.")

def jsonl_to_list(jsonl_path: str, key: str):
    texts = []
    with tqdm(total=1, desc="Loading data") as pbar:
        with open(jsonl_path, 'r', encoding='utf-8') as f_jsonl:
            for line in f_jsonl:
                data = json.loads(line.strip())
                texts.append(data[key])
        pbar.update(1)
    return texts

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
    gen_out = model.generate(**model_inputs, **generate_kwargs)
    gen_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)  # 디코딩을 호출하여 토큰 ID 목록을 문자열 목록으로 변환
    return gen_text  # 토큰 ID의 문자열 목록

def batch_loop(instructions, batch_size, tokenizer, model):
    outputs = []
    for i in tqdm(range(0, 5, batch_size)):  # len(instructions)
        batch = instructions[i: i + batch_size]
        gen_text = process_batch(batch, tokenizer, model)
        outputs.extend(gen_text)
    post_processed_output = [output.replace(inst, "").replace("$}}%>", "").replace("$}}%-", "").replace("$}}%", "").replace("#", "").strip() for inst, output in zip(instructions, outputs)]
    return post_processed_output

# 테스트 데이터 로드
test_file_path = "/data/jaesunghwang/sum_termproject/data/test_data_with_256_instruction.jsonl"
texts = jsonl_to_list(test_file_path, "document")

# 입력 준비
instructions = prepare_inputs(texts)

# 배치 크기 설정
batch_size = 1

# 배치 루프 실행
summaries = batch_loop(instructions, batch_size, tokenizer, model)

# 결과 저장
test_results = [{"summary": summary} for summary in summaries][:5]

# 결과를 JSON Lines (JSONL) 파일로 저장
output_file_path = "/data/jaesunghwang/sum_termproject/results/test_summary_results_lora10.jsonl"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for result in test_results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

# 요약 결과 출력 (필요시 주석 해제)
# for result in test_results:
#     print(f"Summary: {result['summary']}")
