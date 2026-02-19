import os, json

import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F
from tqdm import tqdm
import re


model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
# model_name = "deepseek-ai/deepseek-math-7b-instruct"
device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto",
# ).eval()

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def evaluate(prompt, max_new_tokens=2048, do_sample=True, temperature=0.6, num_return_sequences=10):
#     messages = [
#     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt} ]

#     text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=max_new_tokens,
#         num_return_sequences=num_return_sequences,
#         do_sample=do_sample,
#         temperature=temperature
#     )
#     # Calculate input length for slicing
#     input_length = model_inputs.input_ids.shape[1]

#     # Slice generated output to exclude the prompt
#     generated_ids = [output_ids[input_length:] for output_ids in generated_ids]

#     responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
#     return responses

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

# prm_model_name = "Qwen/Qwen2.5-Math-PRM-7B"
# prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
# prm_model = AutoModel.from_pretrained(
#     prm_model_name,
#     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#     device_map=device,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# ).eval()

# prm800_model_name = "Qwen/Qwen2.5-Math-7B-PRM800K"
# prm800_tokenizer = AutoTokenizer.from_pretrained(prm800_model_name, trust_remote_code=True)
# prm800_model = AutoModel.from_pretrained(
#     prm800_model_name,
#     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#     device_map=device,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# ).eval()

prmdpsk_model_name = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
prmdpsk_tokenizer = AutoTokenizer.from_pretrained(prmdpsk_model_name, trust_remote_code=True)
prmdpsk_model = AutoModel.from_pretrained(
    prmdpsk_model_name,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

prm_shepherd_model_name = "peiyi9979/math-shepherd-mistral-7b-prm"
prm_shepherd_tokenizer = AutoTokenizer.from_pretrained(prm_shepherd_model_name, trust_remote_code=True)
prm_shepherd_model = AutoModel.from_pretrained(
    prm_shepherd_model_name,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map=device,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()


def step_rewards_cal_function(user_query, solution_step_list, model, tokenizer):

    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": user_query,
        "response": solution_step_list
    }

    messages = [
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)
    return step_reward[0]

def step_rewards_cal_function2(user_query, solution_step_list, model, tokenizer):
    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id,minus_tag_id]

    single_step_score = []
    conversation = []
    for k in range(len(solution_step_list)):
        if k == 0:
            text = user_query + " " + solution_step_list[0]
        else:
            text = solution_step_list[k]
        conversation.append({"content": text, "role": "user"})
        conversation.append({"content": "+", "role": "assistant"})

        input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(input_ids)[0][:, -3, candidate_tokens] #simple version, the +/- is predicted by the '-3' position
            scores = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
            # print(scores)
            single_step_score.append(scores[0].detach().to('cpu').item())
    return single_step_score

def step_rewards_cal_function3(user_query, solution_step_list, model, tokenizer):
    single_step_score = []
    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
    
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902

    output = " ки\n".join(solution_step_list) + " ки\n"
    input_for_prm = f"{user_query} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(model.device)

    with torch.no_grad():
        logits = model(input_id)[0][:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        step_scores = scores[input_id == step_tag_id].cpu().tolist()
        single_step_score.append(step_scores)
    return single_step_score[0]

def step_rewards_cal_function_umprm(user_query, solution_step_list, model, tokenizer):
    single_step_score = []
    good_token = '+'
    bad_token = '-'
    step_tag = 'ş'
    
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")
    step_tag_id = tokenizer.encode(f"{step_tag}")[0]

    output = "ş".join(solution_step_list) + "ş"
    input_for_prm = f"{user_query} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to(model.device)

    with torch.no_grad():
        logits = model(input_id)[0][:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        step_scores = scores[input_id == step_tag_id].cpu().tolist()
        single_step_score.append(step_scores)
    return single_step_score[0]

math_questions = []
math_gt_answers = []
with open('eval/data/math500.jsonl', 'r') as f:
    for line in f.readlines():
        line = json.loads(line.strip())
        math_questions.append(line['question'])
        math_gt_answers.append(line['answer'])

with open('outputs/math_qwen2.5_7B_instruct_128_outputs.json', 'r') as f:
    math_qwen_instruct_128_outputs = json.load(f)

with open('outputs/math_gpt4o_sample_128_preds.json', 'r') as f:
    math_gpt4o_sample_128_preds = json.load(f)


total_cost = 0
math_output_samples = []
with open('outputs/math_qwen_output_sample_128_dpsk_shepherd_prm_outputs.jsonl', 'a') as f:
    for i in tqdm(range(129, len(math_questions))):
        output = {}
        # responses = evaluate(math_questions[i], num_return_sequences=64)
        responses = math_qwen_instruct_128_outputs[i]
        # responses = math_gpt4o_sample_128_preds[i]
        prm_step_rewards_list = []
        prm800_step_rewards_list = []
        prmdpsk_step_rewards_list = []
        prm_shepherd_step_rewards_list = []
        for response in responses:
            steps = response.split('\n\n')
            # prm_step_rewards = step_rewards_cal_function(math_questions[i], steps, model=prm_model, tokenizer=prm_tokenizer)
            # prm800_step_rewards = step_rewards_cal_function(math_questions[i], steps, model=prm800_model, tokenizer=prm800_tokenizer)
            prmdpsk_step_rewards = step_rewards_cal_function2(math_questions[i], steps, model=prmdpsk_model, tokenizer=prmdpsk_tokenizer)
            prm_shepherd_step_rewards = step_rewards_cal_function3(math_questions[i], steps, model=prm_shepherd_model, tokenizer=prm_shepherd_tokenizer)
            print(prmdpsk_step_rewards)
            print(prm_shepherd_step_rewards)
            # print(prmdpsk_step_rewards)
            print('----------------')
            # prm_step_rewards_list.append(prm_step_rewards)
            # prm800_step_rewards_list.append(prm800_step_rewards)
            # prmdpsk_step_rewards_list.append(prmdpsk_step_rewards)
            prm_shepherd_step_rewards_list.append(prm_shepherd_step_rewards)

        output['question'] = math_questions[i]
        output['gold_answer'] = math_gt_answers[i]
        output['prediction'] = responses
        # output['prm_step_rewards'] = prm_step_rewards_list
        # output['prm800_step_rewards'] = prm800_step_rewards_list
        output['prmdpsk_step_rewards'] = prmdpsk_step_rewards_list
        output['prm_shepherd_step_rewards'] = prm_shepherd_step_rewards_list

        math_output_samples.append(output)
        f.write(json.dumps(output, ensure_ascii=False) + '\n')
