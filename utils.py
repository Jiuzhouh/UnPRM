import torch.nn.functional as F
from sympy import *
from pydantic import BaseModel
import random
from typing import Optional, Dict, Any, List, Type
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from tqdm import tqdm
import math
import numpy as np
import re

class ag_step(BaseModel):
    question: str = ""
    cur_step: str = ""
    action: str = ""
    action_input: str = ""
    previous_steps: str = ""
    value: Optional[float] = -100
    ans: str = ""
    num_step_tokens:list[int]=[]
    step_prob:float=0
    step_token_probs:tuple=()
    q_value:Optional[float] = -100
    final_value:Optional[float] = -100
    step_value: list[dict] = []
    whole_generate: str=""
    ag_prompt:str=""
    cur_step_tokens:tuple=()
    correct:Optional[bool] = False


def get_prompt(node,history,generate_model_name=''):
    prompt=f"Instruction:{node.question}\nResponse: Let’s think step by step.\n{node.previous_steps}"
    example=random.sample(history,1)
    if "gemma" in generate_model_name:
        generate_prompt = [{"role":"user","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations. Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'.After outputting the final answer only once, be sure to stop outputting."},
                        {"role":"assistant","content":"OK, I understand."},
                        {"role":"user","content":example[0]["instruction"]},
                        {"role":"assistant","content":example[0]["response"]},
                        {"role":"user","content":"Instruction: A square has sides of length 10, and a circle centered at one of its vertices has radius 10. What is the area of the union of the regions enclosed by the square and the circle? Express your answer in terms of $\\pi$.\nResponse: Let’s think step by step.\nStep 1:I want to find the area of the shaded region in this picture, where the blue is the square and the red is the circle.\nStep 2:I notice that the circle and the square share a quarter of the circle's area, which is $\\frac{{1}}{{4}}\\pi r^2$, where $r = 10$.\n"},
                        {"role":"assistant","content":"Step 3:So I can subtract that from the sum of the areas of the circle and the square to get the area of the union.\nStep 4:The area of the circle is $\\pi r^2 = 100\\pi$, and the area of the square is $s^2 = 100$, where $s = 10$.\nStep 5:So the area of the union is $100\\pi + 100 - \\frac{{1}}{{4}}100\\pi = 100 + \\frac{{3}}{{4}}100\\pi$.\nStep 6: The final answer is: 100 + \\frac{{3}}{{4}}100\\pi.\n"},
                        {"role":"user","content":prompt}]
    else:
        # generate_prompt = [{"role":"system","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations. Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'.After outputting the final answer only once, be sure to stop outputting."},
        #                 {"role":"user","content":example[0]["instruction"]},
        #                 {"role":"assistant","content":example[0]["response"]},
        #                 {"role":"user","content":"Instruction: A square has sides of length 10, and a circle centered at one of its vertices has radius 10. What is the area of the union of the regions enclosed by the square and the circle? Express your answer in terms of $\\pi$.\nResponse: Let’s think step by step.\nStep 1:I want to find the area of the shaded region in this picture, where the blue is the square and the red is the circle.\nStep 2:I notice that the circle and the square share a quarter of the circle's area, which is $\\frac{{1}}{{4}}\\pi r^2$, where $r = 10$.\n"},
        #                 {"role":"assistant","content":"Step 3:So I can subtract that from the sum of the areas of the circle and the square to get the area of the union.\nStep 4:The area of the circle is $\\pi r^2 = 100\\pi$, and the area of the square is $s^2 = 100$, where $s = 10$.\nStep 5:So the area of the union is $100\\pi + 100 - \\frac{{1}}{{4}}100\\pi = 100 + \\frac{{3}}{{4}}100\\pi$.\nStep 6: The final answer is: 100 + \\frac{{3}}{{4}}100\\pi.\n"},
        #                 {"role":"user","content":prompt}]
        generate_prompt = [{"role":"system","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations. Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly put the final answer within \\boxed{}. After outputting the final answer only once, be sure to stop outputting."},
                            {"role":"user","content":"Instruction: If the lengths of two sides of a right triangle are 5 and 12 units, what is the least possible length, in units, of the third side? Express your answer in simplest radical form."},
                            {"role":"assistant","content":"Response: Let’s think step by step.\nStep 1: I know that the Pythagorean theorem relates the lengths of the sides of a right triangle by the equation a^2 + b^2 = c^2, where c is the hypotenuse and a and b are the legs.\nStep 2: Since I don't know which side is the hypotenuse, I'll try both possibilities and see which one gives me a smaller value for the third side.\nStep 3: If I assume that the hypotenuse is 12, then the other leg must satisfy 5^2 + b^2 = 12^2, or b^2 = 144 - 25 = 119.\nStep 4: Taking the square root of both sides, I get b = sqrt(119), which is already in simplest radical form.\nStep 5: If I assume that the hypotenuse is the unknown side, then it must satisfy 5^2 + 12^2 = c^2, or c^2 = 25 + 144 = 169.\nStep 6: Taking the square root of both sides, I get c = sqrt(169) = 13.\nStep 7: Comparing the two values, I see that sqrt(119) is smaller than 13, since 119 is smaller than 169. The final answer is \\boxed{119}."},
                            {"role":"user","content":"Instruction: A square has sides of length 10, and a circle centered at one of its vertices has radius 10. What is the area of the union of the regions enclosed by the square and the circle? Express your answer in terms of $\\pi$.\nResponse: Let’s think step by step.\nStep 1: I want to find the area of the shaded region in this picture, where the blue is the square and the red is the circle.\nStep 2: I notice that the circle and the square share a quarter of the circle's area, which is $\\frac{{1}}{{4}}\\pi r^2$, where $r = 10$.\n"},
                            {"role":"assistant","content":"Step 3: So I can subtract that from the sum of the areas of the circle and the square to get the area of the union.\nStep 4: The area of the circle is $\\pi r^2 = 100\\pi$, and the area of the square is $s^2 = 100$, where $s = 10$.\nStep 5: So the area of the union is $100\\pi + 100 - \\frac{{1}}{{4}}100\\pi = 100 + \\frac{{3}}{{4}}100\\pi$.\nStep 6: The final answer is \\boxed{100 + \\frac{{3}}{{4}}100\\pi}."},
                            {"role":"user","content":prompt}]
        # generate_prompt = [
        #                     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        #                     {"role": "user", "content": prompt} ]
    return generate_prompt


def remove_unit(text):
    # Remove all LaTeX units like \text{...}
    text = re.sub(r'\\text\s*{\s*[^}]*}', '', text)
    # Remove LaTeX degree, e.g. ^\circ (with optional space after ^)
    text = re.sub(r'\^\s*\\circ', '', text)
    # Remove trailing exponents like ^2, ^3 after number (with or without space)
    text = re.sub(r'(\d+)\s*\^\d+', r'\1', text)
    # Remove trailing units after a number
    text = re.sub(r'(\d+)\s*(cm\^2|cm|meters|feet|inches|seconds|cents|gallons|beakers|cubic|calories|customers|degrees|nickels|daps|inches/second|°|/second)\b', r'\1', text)
    unit = [
        "calories", "cm^2", "cm", "customers", "degrees", "nickels", "daps", "gallons",
        "inches", "seconds", "cents", "meters", "beakers", "inches/second", "^\circ",
        "\u00b0", "/second", "cubic", "feet"
    ]
    for item in unit:
        text = re.sub(re.escape(item), '', text)
    # Remove double spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_answer(text):
    if 'boxed' in text:
        ans = text.split('boxed')[-1]
        if not ans:
            return ""
        if ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        extract_ans = remove_unit(a)
        if 'frac' in extract_ans and '{{' in extract_ans:
            extract_ans = extract_ans.replace('{{', '{').replace('}}', '}')
    else:
        split_ans = text.split('The final answer is')
        if len(split_ans) == 1 or not split_ans[-1]:
            split_ans = text.split('the final answer is')
            if len(split_ans) == 1 or not split_ans[-1]:
                return "bad_answer"
        ans = split_ans[-1].strip()
        extract_ans = ans[1:].strip() if ans and ans[0] == ":" else ans
        extract_ans = extract_ans.replace("<|eot_id|>", "")
        extract_ans = extract_ans.replace("<|start_header_id|>", "")
        extract_ans = extract_ans.split('I hope it is correct')[-1]
        if not extract_ans:
            return "bad_answer"
        extract_ans = remove_unit(extract_ans)
        extract_ans = extract_ans.split('.\n')[0]
        if extract_ans and extract_ans[-1] == '.':
            extract_ans = extract_ans[:-1]
        if 'frac' in extract_ans and '{{' in extract_ans:
            extract_ans = extract_ans.replace('{{', '{').replace('}}', '}')
        if len(extract_ans) > 1 and extract_ans[0] == "$":
            extract_ans = extract_ans[1:]
        if len(extract_ans) > 1 and extract_ans[-1] == "$":
            extract_ans = extract_ans[:-1]
        if len(extract_ans) >= 2 and extract_ans[:2] == "\\(":
            extract_ans = extract_ans[2:]
        if len(extract_ans) >= 2 and extract_ans[-2:] == "\\)":
            extract_ans = extract_ans[:-2]
        if len(extract_ans) >= 200:
            return "bad_answer"

    return extract_ans


def get_sc_final_answer(solutions_node,answer_dict,part):
    for i,node in enumerate(solutions_node):
        answer=extract_answer(node.solution) if not part else extract_answer(node.cur_step)
        if answer in answer_dict:
            answer_dict[answer]+=1
        else:
            answer_dict[answer]=1
    return answer_dict


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def add_tokenizer(model, tokenizer):
    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.unk_token_id = tokenizer.unk_token_id


def get_model_name(model_dir):
    model_name_list=["llama-3","Llama-3.1","qwen2","deepseek","llemma","phi3.5","phi-3","Qwen2.5","Qwen3","gemma-2-2b","gemma-2-9b","Ministral-8B-Instruct","Mistral-7B-Instruct-v0.3"]
    model_name = next((name for name in model_name_list if name in model_dir), None)
    return model_name


def get_dataset_name(data_dir):
    data_name_list=["test500","gaokao23","AIME","olympiadbench","minerva"]
    data_name = next((name for name in data_name_list if name in data_dir), None)
    return data_name


def tokenize_verify_input(verify_prompt, next_step, verify_tokenizer):
    input_ids_list = []
    for i in range(len(verify_prompt)):
        tokenized_next_step = verify_tokenizer(next_step[i], return_tensors="pt", max_length=8000, truncation=True).input_ids[0]
        tokenized_state = verify_tokenizer(verify_prompt[i], padding=False, max_length=verify_tokenizer.model_max_length - len(tokenized_next_step), truncation=True, return_tensors="pt").input_ids[0]
        input_ids_list.append(torch.cat((tokenized_state, tokenized_next_step), dim=0))
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=verify_tokenizer.pad_token_id).to("cuda")
    input_ids = dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(verify_tokenizer.pad_token_id),
    )
    return input_ids


def verify_solution(nodes,verify_llm, verify_tokenizer,num_labels,part,batch_size=1):
    select_steps_list=[]
    question_list=[]
    step_length_list=[]
    next_steps=[]
    for node in nodes:
        solution=node.cur_step
        if extract_answer(solution)=="" or extract_answer(solution)=="bad_answer":
            node.final_value=-1
            node.step_value=[-1.0]
            step_length_list.append(0)
            continue
        severed_solution=solution.split("Step")
        select_steps="\n" if not part else node.previous_steps
        severed_solution=severed_solution[1:] if severed_solution[0]=="" else severed_solution
        select_steps_list.append(select_steps)
        step_length_list.append(len(severed_solution))
        for i in range(len(severed_solution)):
            question_list.append(node.question)
            next_steps.append(f"Step"+severed_solution[i])
            select_steps+=next_steps[-1]
            if i!=len(severed_solution)-1:
                select_steps_list.append(select_steps)
    value_list=[]
    for indx in range(math.ceil(len(select_steps_list)/batch_size)):
        value_list+=verify(question_list[indx*batch_size:(indx+1)*batch_size], verify_llm, verify_tokenizer, select_steps_list[indx*batch_size:(indx+1)*batch_size], next_steps[indx*batch_size:(indx+1)*batch_size],num_labels)
    step_sum=0
    for i in range(len(nodes)):
        if step_length_list[i]!=0:
            values=value_list[step_sum:step_sum+step_length_list[i]]
            nodes[i].step_value=values
            nodes[i].final_value=min(values) if num_labels!=1 else max(values)
            step_sum+=step_length_list[i]
    return nodes



def verify(question, verify_llm, verify_tokenizer, select_steps_list, next_steps,num_labels):
    prompt = "### Instruct:\nThe steps in 'Selected steps' are the correct problem-solving steps of the problem, while the steps in 'Next step:' are the next problem-solving steps generated by an AI agent based on the steps in 'Selected steps:'.You need to rate the step in 'Next step:' based on it`s usefulness and correctness.\n\n"
    next_step_list = ["Next step:" + next_step for next_step in next_steps]
    for item in next_step_list:
        if "I can’t help with this question." in item:
            return [0.0]
    verify_prompt = [prompt + "Problem:" + question[i] + "\n" +"Select steps:"+select_steps_list[i] for i in range(len(select_steps_list))]
    input_ids = tokenize_verify_input(verify_prompt, next_step_list, verify_tokenizer)
    with torch.no_grad():
        output = verify_llm(**input_ids)
    pred = F.softmax(output.logits, dim=-1)
    return [float(pred[i][0]) for i in range(len(pred))]

def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

def cal_uncertainty_entropy(token_logprobs, temperature=5):
    logits_softmax = temperature_scaling(token_logprobs, temperature)
    uncertainty = np.sum(-np.log(logits_softmax)*logits_softmax)
    # uncertainty = entropy(logits_softmax, base=2)
    return uncertainty

def cal_uncertainty_log_sum(token_logprobs, temperature=5):
    logits_softmax = temperature_scaling(token_logprobs, temperature)
    uncertainty = np.sum(-np.log(logits_softmax))
    return uncertainty