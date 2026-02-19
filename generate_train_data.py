import os

import numpy as np
from sympy import *
from vllm import LLM, SamplingParams
from statistics import mean
import random
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from math_utils import math_is_equiv
import argparse
from utils import ag_step, extract_answer, get_model_name, get_prompt, cal_uncertainty_entropy, cal_uncertainty_log_sum
import time
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--model_dir", type=str, default='')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--generate", action= "store_true")
    parser.add_argument("--uncertainty_driven_generate", action= "store_true")
    parser.add_argument("--get_label", action= "store_true")
    parser.add_argument("--synthetic_data_dir", type=str, default='')
    parser.add_argument("--low_level", action= "store_true")
    parser.add_argument("--use_adaptive_binary", action= "store_true")
    parser.add_argument("--sequential_search", action= "store_true")
    parser.add_argument("--use_uncertainty_driven", action= "store_true")
    parser.add_argument("--use_uncertainty_delta_driven", action= "store_true")
    parser.add_argument("--uncertainty_method", type=str, default="entropy")
    parser.add_argument("--uncertainty_driven_data_generation", action= "store_true")
    return parser.parse_args()


def eval_answer(pred_text,ground_truth):
    import signal
    timeout_seconds = 10*60
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    try:
        pred=extract_answer(pred_text)
        if pred=="bad_answer":
            return False
        # print(pred,"===============",ground_truth)
        result = math_is_equiv(pred,ground_truth)
    except TimeoutError as exc:
        result = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
    return result


def extract_token_logprobs(token_logprob_dicts, token_ids):
    tokens = []
    logprobs = []
    for i, token_id in enumerate(token_ids):
        d = token_logprob_dicts[i]
        tokens.append(d[token_id].decoded_token)
        logprobs.append(d[token_id].logprob)
        
    return (tokens, logprobs)


def generate_numi_steps_test(node, generate_tokenizer, generate_llm, test_n, numi,temperature,top_p,ori_prompt,max_tokens=None,gemma=False):
    stop = ["I hope it is correct.", "ASSISTANT:", "ASSISTANT", "assistant", f"Step {numi + 2}", f"step {numi + 2}", f"Step{numi + 2}", f"step{numi + 2}","<|eot_id|>","<|start_header_id|>","<|endoftext|>","end{document}"]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=-1,
        top_p=top_p,
        max_tokens=800 if numi>0 else 4000,
        n=test_n,
        stop=stop,
        seed=69,
        logprobs=1
    )
    sampling_params.max_tokens=max_tokens if max_tokens is not None else sampling_params.max_tokens
    prompt = f"Instruction:{node.question}\nResponse: Let’s think step by step.\n{node.previous_steps}"
    if ori_prompt is None:
        if gemma:
            generate_prompt = [{"role":"user","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations. Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'.After outputting the final answer only once, be sure to stop outputting."},
                            {"role":"assistant","content":"OK, I understand."},
                            {"role":"user","content":"Instruction: If the lengths of two sides of a right triangle are 5 and 12 units, what is the least possible length, in units, of the third side? Express your answer in simplest radical form."},
                            {"role":"assistant","content":"Response: Let’s think step by step.\nStep 1:I know that the Pythagorean theorem relates the lengths of the sides of a right triangle by the equation a^2 + b^2 = c^2, where c is the hypotenuse and a and b are the legs.\nStep 2:Since I don't know which side is the hypotenuse, I'll try both possibilities and see which one gives me a smaller value for the third side.\nStep 3:If I assume that the hypotenuse is 12, then the other leg must satisfy 5^2 + b^2 = 12^2, or b^2 = 144 - 25 = 119.\nStep 4:Taking the square root of both sides, I get b = sqrt(119), which is already in simplest radical form.\nStep 5:If I assume that the hypotenuse is the unknown side, then it must satisfy 5^2 + 12^2 = c^2, or c^2 = 25 + 144 = 169.\nStep 6:Taking the square root of both sides, I get c = sqrt(169) = 13.\nStep 7:Comparing the two values, I see that sqrt(119) is smaller than 13, since 119 is smaller than 169. The final answer is 119\n"},
                            {"role":"user","content":"Instruction: A square has sides of length 10, and a circle centered at one of its vertices has radius 10. What is the area of the union of the regions enclosed by the square and the circle? Express your answer in terms of $\\pi$.\nResponse: Let’s think step by step.\nStep 1:I want to find the area of the shaded region in this picture, where the blue is the square and the red is the circle.\nStep 2:I notice that the circle and the square share a quarter of the circle's area, which is $\\frac{{1}}{{4}}\\pi r^2$, where $r = 10$.\n"},
                            {"role":"assistant","content":"Step 3:So I can subtract that from the sum of the areas of the circle and the square to get the area of the union.\nStep 4:The area of the circle is $\\pi r^2 = 100\\pi$, and the area of the square is $s^2 = 100$, where $s = 10$.\nStep 5:So the area of the union is $100\\pi + 100 - \\frac{{1}}{{4}}100\\pi = 100 + \\frac{{3}}{{4}}100\\pi$.\nStep 6: The final answer is: 100 + \\frac{{3}}{{4}}100\\pi.\n"},
                            {"role":"user","content":prompt}]
        else:
            # generate_prompt = [{"role":"system","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations. Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'. After outputting the final answer only once, be sure to stop outputting."},
            #                 {"role":"user","content":"Instruction: If the lengths of two sides of a right triangle are 5 and 12 units, what is the least possible length, in units, of the third side? Express your answer in simplest radical form."},
            #                 {"role":"assistant","content":"Response: Let’s think step by step.\nStep 1:I know that the Pythagorean theorem relates the lengths of the sides of a right triangle by the equation a^2 + b^2 = c^2, where c is the hypotenuse and a and b are the legs.\nStep 2:Since I don't know which side is the hypotenuse, I'll try both possibilities and see which one gives me a smaller value for the third side.\nStep 3:If I assume that the hypotenuse is 12, then the other leg must satisfy 5^2 + b^2 = 12^2, or b^2 = 144 - 25 = 119.\nStep 4:Taking the square root of both sides, I get b = sqrt(119), which is already in simplest radical form.\nStep 5:If I assume that the hypotenuse is the unknown side, then it must satisfy 5^2 + 12^2 = c^2, or c^2 = 25 + 144 = 169.\nStep 6:Taking the square root of both sides, I get c = sqrt(169) = 13.\nStep 7:Comparing the two values, I see that sqrt(119) is smaller than 13, since 119 is smaller than 169. The final answer is 119\n"},
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
            #                 {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            #                 {"role": "user", "content": prompt}]
            # generate_prompt = [{"role":"system","content":"You are a powerful agent with broad math knowledge and good at accurate calculation on math equations. Below is an instruction that describes a task. Continue to finish the response that appropriately completes the request Within a maximum of 40 steps. When outputting each step, mark the sequence number of each step at the beginning, and explicitly state the final answer after the final step following the format 'The final answer is:'.After outputting the final answer only once, be sure to stop outputting."},
            #                 {"role":"user","content":prompt}]
    else:
        generate_prompt=ori_prompt

    generate_input = generate_tokenizer.apply_chat_template(generate_prompt,tokenize=False,add_generation_prompt=True,)
    # print(generate_input)
    outputs = generate_llm.generate(generate_input, sampling_params, use_tqdm=False)
    # print(outputs)
    # with open("log.txt","a") as file:
    #     file.write(str(outputs) + '\n')
    # outputs = generate_llm.generate(generate_input, use_tqdm=False)
    
    outputs_text,outputs_probs,outputs_token_probs,num_output_tokens,cur_step_tokens=[],[],[],[],[]
    for output in outputs:
        for idx in range(test_n):
            try:
                text=output.outputs[idx].text
                # print(text)
                # print('--------------------------------')
            except IndexError:
                continue
            if len(text)<10:
                continue
            outputs_text.append(text.strip())
            outputs_probs.append(output.outputs[idx].cumulative_logprob if output.outputs[idx].cumulative_logprob is not None else 0.0)
            outputs_token_probs.append(extract_token_logprobs(output.outputs[idx].logprobs, output.outputs[idx].token_ids))
            num_output_tokens.append([len(output.prompt_token_ids),len(output.outputs[idx].token_ids)])
            cur_step_tokens.append(output.outputs[idx].token_ids)
    return [ag_step(question=node.question, previous_steps=node.previous_steps, cur_step=outputs_text[i], num_step_tokens=num_output_tokens[i], step_prob=outputs_probs[i], step_token_probs=outputs_token_probs[i], cur_step_tokens=cur_step_tokens[i], ans=node.ans) for i in range(len(outputs_text))]

class generate_prm_data:

    def __init__(self,args)-> None:
        self.data_dir=args.data_dir
        self.save_dir=args.save_dir
        self.model_dir=args.model_dir
        self.seed=args.seed
        self.start=args.start
        self.end=args.end
        self.synthetic_data_dir=args.synthetic_data_dir
        self.low_level=args.low_level
        self.use_adaptive_binary=args.use_adaptive_binary
        self.sequential_search=args.sequential_search
        self.use_uncertainty_driven=args.use_uncertainty_driven
        self.use_uncertainty_delta_driven=args.use_uncertainty_delta_driven
        self.uncertainty_method=args.uncertainty_method
        self.uncertainty_driven_data_generation=args.uncertainty_driven_data_generation

    def load_data(data_dir):
        data=[]
        with open(data_dir,"r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_label(self):
        start_time=time.time()
        random.seed(self.seed)
        if self.sequential_search:
            mode="sequential_search"
            static_sample=True
        elif self.use_adaptive_binary:
            mode="binary_pro_search"
            static_sample=False
        elif self.use_uncertainty_driven:
            if self.uncertainty_method == "log_sum":
                mode="uncertainty_driven_search (log_sum)"
            elif self.uncertainty_method == "entropy":
                mode="uncertainty_driven_search (entropy)"
            else:
                raise ValueError("Unknown uncertainty method")
            static_sample=False
        elif self.use_uncertainty_delta_driven:
            if self.uncertainty_method == "log_sum":
                mode="uncertainty_delta_driven_search (log_sum)"
            elif self.uncertainty_method == "entropy":
                mode="uncertainty_delta_driven_search (entropy)"
            else:
                raise ValueError("Unknown uncertainty method")
            static_sample=False
        else:
            mode="binary_search"
            static_sample=True
        print(f"搜索算法:{mode}; 静态采样:{static_sample}\n")
        # llm_dir_list=["meta-llama/Llama-3.1-8B-Instruct"]
        llm_dir_list=[self.model_dir]
        gpu_memory_utilization_list=[0.5,0.45]
        llm_list=[LLM(
                    model=llm_dir_list[i],
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_memory_utilization_list[i],
                    max_model_len=2048,
                    swap_space=16,
                    seed=69,
                ) for i in range(len(llm_dir_list))]
        tokenizer_list=[AutoTokenizer.from_pretrained(model_dir) for model_dir in llm_dir_list]
        synthetic_data=[]
        with open(self.synthetic_data_dir,"r") as f:
            for line in f:
                synthetic_data.append(json.loads(line))
        total_steps=0
        history=[]
        total_sampled,total_sample_tokens=0,0
        for i,data in tqdm(enumerate(synthetic_data),total=min(len(synthetic_data),self.end),desc="Main process"):
            if self.start<=i<self.end:
                question=data["question"]
                answer=data["ground_truth"]
                if self.uncertainty_driven_data_generation:
                    solutions=data["solutions_uncertainty_driven"]            
                else:
                    solutions=data["solutions_similarity_driven"]
                step_node=ag_step(question=question,previous_steps="",ans=answer)
                if static_sample:
                    sample_num_n=48
                    solved_list,perplexity_list,generate_tokens,_=self.verify_step(step_node,llm_list,tokenizer_list,sample_num_n,history)
                    total_sampled+=sample_num_n
                    total_sample_tokens+=generate_tokens
                else:
                    sample_num_n=72
                    solved_list,perplexity_list,generate_tokens,generate_nodes=self.verify_step(step_node,llm_list,tokenizer_list,sample_num_n,history)
                    total_sampled+=sample_num_n
                    total_sample_tokens+=generate_tokens
                    if sum(solved_list)<=10:
                        print(f"问题{i}跳过")
                        continue
                    solved_list=[0 for _ in range(len(generate_nodes))]
                    perplexity_list=[0 for _ in range(len(generate_nodes))]
                    for i in range(len(generate_nodes[0])):
                        for j in range(len(generate_nodes)):
                            if generate_nodes[j][i].correct:
                                solved_list[j]+=1
                        if sum(solved_list)>=10:
                            sample_num_n=math.ceil(i/8)*8
                            sample_num_n=max(16,sample_num_n)
                            break
                    solved_list,perplexity_list,generate_tokens,_=self.verify_step(step_node,llm_list,tokenizer_list,sample_num_n,history)
                    total_sampled+=sample_num_n
                    total_sample_tokens+=generate_tokens
                print(f"sampling {sample_num_n} candidates")
                for solution in tqdm(solutions,total=len(solutions),desc="Solution process"):
                    # step_list=solution["solution"].split("Step")
                    step_list = solution['solution'].split('\nStep')
                    step_list = step_list[1:] if step_list[0]=="" else step_list
                    step_list[0] = step_list[0].split('Step')[1]
                    tokens, logprobs = solution['step_token_logprobs']
                    steps, step_probs = self.split_by_step(tokens, logprobs)
                    try:
                        assert len(step_list) == len(step_probs), "step_list and step_probs not the same length"
                    except:
                        continue
                    
                    final_data={"problem":question,"ground_truth_answer":answer,"total_steps":len(step_list),"solved_perplexity":perplexity_list[0],"sample_num":sample_num_n}
                    total_steps+=len(step_list)
                    previous_steps=""
                    step_node_list=[]
                    final_solved_perplexity_list=[[] for _ in range(len(step_list))]
                    for step_indx in range(len(step_list)-1):
                        previous_steps+="Step"+step_list[step_indx] if step_list[step_indx] else ""+step_list[step_indx]
                        step_node_list.append(ag_step(question=final_data["problem"],previous_steps=previous_steps,step_token_probs=step_probs[step_indx],ans=final_data["ground_truth_answer"]))
                    previous_steps=""
                    if not solution["label"]:
                        result = self.find_error_step(final_data,0,len(step_node_list)-1,step_node_list,llm_list,tokenizer_list,sample_num_n,history,final_solved_perplexity_list)
                        first_error_step = result['first_error_step']
                        solved_perplexity_list = result['final_solved_perplexity_list']
                        sampled = result['total_sampled']
                        generate_tokens = result['total_sample_tokens']
                        num_sample_steps = result['num_sample_steps'] 
                        error_step_uncertainty_rank = result['error_step_uncertainty_rank']
                        step_uncertainties = result['step_uncertainties']
                        
                        total_sampled+=sampled
                        total_sample_tokens+=generate_tokens
                        final_data["final_solved_perplexity_list"]=solved_perplexity_list
                        final_data["num_sample_steps"]=num_sample_steps
                        final_data["error_step_uncertainty_rank"]=error_step_uncertainty_rank
                        final_data["step_uncertainties"]=step_uncertainties
                    for indx,step in enumerate(step_list):
                        final_data[f"state{indx}"]="Problem: "+final_data["problem"]+"\nSelected steps:\n"+previous_steps
                        if indx==len(step_list)-1:
                            final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":solution["label"]}]
                            continue
                        previous_steps+="Step"+step if step else ""+step
                        if solution["label"]:
                            final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":solution["label"]}]
                        else:
                            if indx>=first_error_step:
                                final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":0}] if indx==first_error_step else [{"text":"Step"+step,"rating":0}]
                            else:
                                final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":1}]

                    finish_time=time.time()
                    final_data["total_sampled"]=total_sampled
                    final_data["total_sample_tokens"]=total_sample_tokens
                    final_data["total_use_time"]=finish_time-start_time
                    with open(self.save_dir,"a") as file:
                        file.write(json.dumps(final_data) + '\n')
        print(f"搜索算法:{mode}, 共生成token数:{total_sample_tokens}, 共采样次数:{total_sampled}, 共用时:{finish_time-start_time}\n")

    def split_by_step(self, tokens, probs):
        result_tokens = []
        result_probs = []
        current_tokens = []
        current_probs = []
        for token, prob in zip(tokens, probs):
            if token == 'Step':
                if current_tokens:
                    result_tokens.append(current_tokens)
                    result_probs.append(current_probs)
                current_tokens = [token]
                current_probs = [prob]
            else:
                current_tokens.append(token)
                current_probs.append(prob)
        if current_tokens:
            result_tokens.append(current_tokens)
            result_probs.append(current_probs)
        return result_tokens, result_probs


    def find_error_step(self,final_data,left,right,step_node_list,llm_list,tokenizer_list,sample_num_n,history,final_solved_perplexity_list):
        c=0
        # solved_perplexity=final_data["llama3.1_solved_perplexity"]+final_data["qwen2_solved_perplexity"]
        solved_perplexity=final_data["solved_perplexity"]
        acc_threshold=solved_perplexity
        first_search=True
        total_sampled,total_sample_tokens=0,0
        if self.sequential_search:
            left=len(step_node_list)
            for i in range(len(step_node_list)):
                _,solved_perplexity_list,generate_tokens,_=self.verify_step(step_node_list[i],llm_list,tokenizer_list,sample_num_n,history)
                total_sampled+=sample_num_n
                total_sample_tokens+=generate_tokens
                final_solved_perplexity_list[i]=solved_perplexity_list
                solved=sum(solved_perplexity_list)
                c+=1
                if solved<acc_threshold:
                    left=i
                    print(f"\nfind error step{left}, sample {c} steps\n")
                    break

            return {
                "first_error_step": left,
                "final_solved_perplexity_list": final_solved_perplexity_list,
                "total_sampled": total_sampled,
                "total_sample_tokens": total_sample_tokens,
                "num_sample_steps": c,
                "error_step_uncertainty_rank": None,
                "step_uncertainties": None
            }
        
        elif self.use_uncertainty_driven:
            if self.uncertainty_method == "log_sum":
                uncertainty_func = cal_uncertainty_log_sum
            elif self.uncertainty_method == "entropy":
                uncertainty_func = cal_uncertainty_entropy
            else:
                raise ValueError("Unknown uncertainty method")
            # 1. Compute uncertainty for each step
            step_probs = [node.step_token_probs for node in step_node_list]

            step_uncertainties = [
                (i, uncertainty_func(prob, 5))
                for i, prob in enumerate(step_probs)
            ]
            
            # 2. Sort steps by descending uncertainty
            step_uncertainties.sort(key=lambda x: x[1], reverse=True)
            
            # 3. Iterate steps by uncertainty
            last_rank = None
            last_unc = None
            for rank, (i, unc) in enumerate(step_uncertainties):
                _, solved_perplexity_list, generate_tokens, _ = self.verify_step(
                    step_node_list[i],
                    llm_list,
                    tokenizer_list,
                    sample_num_n,
                    history
                )
                total_sampled += sample_num_n
                total_sample_tokens += generate_tokens
                final_solved_perplexity_list[i] = solved_perplexity_list
                solved = sum(solved_perplexity_list)
                c += 1
                last_rank = rank
                last_unc = unc
                if solved < acc_threshold:
                    print(f"\nFound error at step {i} (uncertainty rank {rank}, value={unc:.4f}), after sampling {c} steps\n")
                    return {
                        "first_error_step": i,
                        "final_solved_perplexity_list": final_solved_perplexity_list,
                        "total_sampled": total_sampled,
                        "total_sample_tokens": total_sample_tokens,
                        "num_sample_steps": c,
                        "error_step_uncertainty_rank": rank,
                        "step_uncertainties": step_uncertainties
                    }
            # If no error found, then the last step is the error step
            if last_unc is not None:
                last_unc_str = f"{last_unc:.4f}"
            else:
                last_unc_str = "N/A"
            print(f"\nFound error at step {len(step_node_list)} (uncertainty diff rank {last_rank}, value={last_unc_str}), after sampling {c} steps\n")
            return {
                "first_error_step": len(step_node_list),
                "final_solved_perplexity_list": final_solved_perplexity_list,
                "total_sampled": total_sampled,
                "total_sample_tokens": total_sample_tokens,
                "num_sample_steps": c,
                "error_step_uncertainty_rank": last_rank,
                "step_uncertainties": step_uncertainties
            }

        elif self.use_uncertainty_delta_driven:
            if self.uncertainty_method == "log_sum":
                uncertainty_func = cal_uncertainty_log_sum
            elif self.uncertainty_method == "entropy":
                uncertainty_func = cal_uncertainty_entropy
            else:
                raise ValueError("Unknown uncertainty method")
            
             # 1. Compute uncertainty for each step (not tuples, just the values)
            step_probs = [node.step_token_probs for node in step_node_list]
            step_uncertainty = [uncertainty_func(prob, 5) for prob in step_probs]
            
            # 2. Compute difference in uncertainty between steps
            diff_uncertainty = [
                step_uncertainty[i+1] - step_uncertainty[i] 
                for i in range(len(step_uncertainty) - 1)
            ]
            # For each diff, associate with the 'step' where the jump starts or ends.
            # Here, we'll associate the diff with step i+1 (the later step).
            diff_indices = list(range(1, len(step_uncertainty)))
            
            # 3. Sort steps by descending difference
            diff_uncertainty_with_idx = [
                (idx, diff) for idx, diff in zip(diff_indices, diff_uncertainty)
            ]

            diff_uncertainty_with_idx.sort(key=lambda x: x[1], reverse=True)
            
            # 4. Iterate steps by difference-based ranking
            last_rank = None
            last_diff = None
            for rank, (i, diff) in enumerate(diff_uncertainty_with_idx):
                _, solved_perplexity_list, generate_tokens, _ = self.verify_step(
                    step_node_list[i],
                    llm_list,
                    tokenizer_list,
                    sample_num_n,
                    history
                )
                total_sampled += sample_num_n
                total_sample_tokens += generate_tokens
                final_solved_perplexity_list[i] = solved_perplexity_list
                solved = sum(solved_perplexity_list)
                c += 1
                last_rank = rank
                last_diff = diff
                if solved < acc_threshold:
                    print(f"\nFound error at step {i} (uncertainty diff rank {rank}, value={diff:.4f}), after sampling {c} steps\n")
                    return {
                        "first_error_step": i,
                        "final_solved_perplexity_list": final_solved_perplexity_list,
                        "total_sampled": total_sampled,
                        "total_sample_tokens": total_sample_tokens,
                        "num_sample_steps": c,
                        "error_step_uncertainty_rank": rank,
                        "step_uncertainties": diff_uncertainty_with_idx
                    }
            # If no error found, then the last step is the error step
            if last_diff is not None:
                last_diff_str = f"{last_diff:.4f}"
            else:
                last_diff_str = "N/A"
            print(f"\nFound error at step {len(step_node_list)} (uncertainty diff rank {last_rank}, value={last_diff_str}), after sampling {c} steps\n")
            return {
                "first_error_step": len(step_node_list),
                "final_solved_perplexity_list": final_solved_perplexity_list,
                "total_sampled": total_sampled,
                "total_sample_tokens": total_sample_tokens,
                "num_sample_steps": c,
                "error_step_uncertainty_rank": last_rank,
                "step_uncertainties": diff_uncertainty_with_idx
            }
        
        else:
            while left <= right:
                solved_rating=min(round(5*solved_perplexity/(2*final_data["sample_num"]))*0.2,1) if not self.use_adaptive_binary else round(solved_perplexity*5)
                mid = (right + left)//2
                if self.use_adaptive_binary and first_search and right>=4:
                    if solved_rating<2:
                        mid-=right//4
                    elif solved_rating>=5:
                        mid+=right//4
                _,solved_perplexity_list,generate_tokens,_=self.verify_step(step_node_list[mid],llm_list,tokenizer_list,sample_num_n,history)
                total_sampled+=sample_num_n
                total_sample_tokens+=generate_tokens
                final_solved_perplexity_list[mid]=solved_perplexity_list
                c+=1
                solved=sum(solved_perplexity_list)
                if solved<acc_threshold:
                    right = mid-1
                else:
                    left = mid+1
                first_search=False
            print(f"\nfind error step{left}, sample {c} steps\n")
            return {
                "first_error_step": left,
                "final_solved_perplexity_list": final_solved_perplexity_list,
                "total_sampled": total_sampled,
                "total_sample_tokens": total_sample_tokens,
                "num_sample_steps": c,
                "error_step_uncertainty_rank": None,
                "step_uncertainties": None
            }
        

    def verify_step(self,node,llm_list,tokenizer_list,sample_num,history):
        acc_list, acc_perplexity_list=[],[]
        return_nodes=[]
        for i in range(len(llm_list)):
            generate_prompt = get_prompt(node,history) if len(history)>1 else None
            generate_node=generate_numi_steps_test(node, tokenizer_list[i], llm_list[i], sample_num, -10, temperature=0.8, top_p=1.0, ori_prompt=generate_prompt, max_tokens=4000)
            acc,acc_prob_n,prob_sum_n,generate_tokens=0,0,0,0
            for gen_node in generate_node:
                if eval_answer(gen_node.cur_step,gen_node.ans):
                    gen_node.correct=True
                    acc+=1
                    acc_prob_n+=np.exp(gen_node.step_prob/gen_node.num_step_tokens[1])
                prob_sum_n+=np.exp(gen_node.step_prob/gen_node.num_step_tokens[1])
                generate_tokens+=gen_node.num_step_tokens[1]
            acc_list.append(acc)
            acc_perplexity_list.append(acc_prob_n / prob_sum_n if prob_sum_n != 0 else 0.0)
            return_nodes.append(generate_node)
        return acc_list,acc_perplexity_list,generate_tokens,return_nodes


    def chose_solution(self,node_list,chose_num):
        if len(node_list)<=chose_num:
            print(f"len(node_list)<=chose_num, return all nodes")
            return node_list
        cosine_similarity_dict={}
        chose_id=[random.randint(0, len(node_list)-1)]
        for i in range(len(node_list)):
            cosine_similarity_dict[i]=[self.cosine_similarity(node_list[i],node_list[j]) for j in range(i,len(node_list))]
        for _ in range(chose_num-1):
            min_cosine_similarity,chosen=1000,chose_id[0]
            for a in range(len(node_list)):
                if a in chose_id:
                    continue
                sum_cosine_similarity=0
                for b in chose_id:
                    sum_cosine_similarity+=cosine_similarity_dict[min(b,a)][max(b,a)-min(b,a)-1]
                mean_cosine_similarity=sum_cosine_similarity/len(chose_id)
                if mean_cosine_similarity<min_cosine_similarity:
                    min_cosine_similarity=mean_cosine_similarity
                    chosen=a
            chose_id.append(chosen)

        print("chose_id", chose_id)
        return [node_list[id] for id in chose_id]


    def choose_solution_uncertainty_driven(self, node_list, choose_num, uncertainty_func=cal_uncertainty_entropy, alpha=0.8):
        """
        Select choose_num nodes that are both diverse and uncertain.
        """
        if len(node_list) <= choose_num:
            print(f"u_driven: len(node_list)<=chose_num, return all nodes")
            return node_list

        # STEP 1: Compute solution uncertainty for each node
        solution_uncertainties = []
        for node in node_list:
            tokens, logprobs = node.step_token_probs
            steps, step_probs = self.split_by_step(tokens, logprobs)
            step_uncertainty = [uncertainty_func(prob, 5) for prob in step_probs]
            diff_uncertainty = [
                step_uncertainty[i+1] - step_uncertainty[i] 
                for i in range(len(step_uncertainty) - 1)
            ]
            # You may choose mean(abs) or another metric
            sol_unc = np.mean([np.abs(u) for u in diff_uncertainty])
            solution_uncertainties.append(sol_unc)

        # Normalize uncertainties to [0, 1]
        min_unc, max_unc = min(solution_uncertainties), max(solution_uncertainties)
        if max_unc == min_unc:
            norm_uncertainties = [0.5] * len(solution_uncertainties)  # fallback
        else:
            norm_uncertainties = [
                (u - min_unc) / (max_unc - min_unc)
                for u in solution_uncertainties
            ]
        
        # STEP 2: Prepare cosine similarity matrix (full matrix for easier lookup)
        N = len(node_list)
        cosine_sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                sim = self.cosine_similarity(node_list[i], node_list[j])
                cosine_sim_matrix[i, j] = sim
                cosine_sim_matrix[j, i] = sim  # matrix is symmetric

        # STEP 3: Selection loop
        # chosen_ids = [random.randint(0, N-1)]
        chosen_ids = [np.argmax(norm_uncertainties)]
        while len(chosen_ids) < choose_num:
            best_score = -float('inf')
            best_candidate = None
            for candidate in range(N):
                if candidate in chosen_ids:
                    continue
                # Mean cosine similarity to chosen set
                mean_sim = np.mean([cosine_sim_matrix[candidate, c] for c in chosen_ids])
                # We want LOW similarity (diversity), so use (1 - mean_sim)
                # Combine with normalized uncertainty
                score = alpha * norm_uncertainties[candidate] + (1 - alpha) * (1 - mean_sim)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                print("WARNING: No valid candidate found; returning current chosen_ids")
                break
            chosen_ids.append(best_candidate)
        print("u_driven_chosen_ids", chosen_ids)
        return [node_list[i] for i in chosen_ids]


    def cosine_similarity(self, node1, node2):
        array1 = np.array(node1.cur_step_tokens)
        array2 = np.array(node2.cur_step_tokens)
        min_len = min(len(array1), len(array2))
        array1 = array1[:min_len]
        array2 = array2[:min_len]
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)
        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity


    def generate_train_solution(self):
        random.seed(self.seed)
        history=[]
        llm = LLM(
            model=self.model_dir,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            seed=69,
        )
        generator_name=get_model_name(self.model_dir)
        tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        data_list=[]
        if self.low_level:
            level_dict={1:500,2:500,3:0,4:0,5:0}
        else:
            level_dict={1:0,2:0,3:500,4:1000,5:1000}
        ori_data_dict={}
        with open(self.data_dir,"r") as f:
            i=0
            for line in f:
                if 5000<i<8000:
                    item=json.loads(line)
                    if level_dict[item["level"]]:
                        data_list.append(item)
                        level_dict[item["level"]]-=1
                i+=1
        random.shuffle(data_list)
        for indx,data in tqdm(enumerate(data_list),total=len(data_list),desc="Main process"):
            if self.start<=indx<self.end:
                if data["problem"] in ori_data_dict:
                    save_data=ori_data_dict[data["problem"]]
                    with open(self.save_dir,"a") as file:
                        file.write(json.dumps(save_data) + '\n')
                    continue
                t_solution_list,f_solution_list=[],[]
                question_node=ag_step(question=data["problem"],ans=data["answer"])
                save_data={"question":question_node.question,"ground_truth":question_node.ans,"level":data["level"],"solutions":[]}
                solved=0                
                generate_prompt=get_prompt(question_node,history) if len(history)>1 else None
                sample_num=min(2**(data["level"]+3),64)
                solution_nodes=generate_numi_steps_test(question_node, tokenizer, llm, sample_num, -10, temperature=0.8, top_p=1, ori_prompt=generate_prompt, max_tokens=2048)
                for node in solution_nodes:
                    generate_answer=extract_answer(node.cur_step)
                    if generate_answer=="" or generate_answer=="bad_answer":
                        continue
                    if math_is_equiv(generate_answer,question_node.ans):
                        node.final_value=1
                        solved+=1
                        history.append({"instruction":f"Instruction:{node.question}\nResponse: Let’s think step by step.\n{node.previous_steps}","response":node.cur_step})
                        t_solution_list.append(node)
                        history=history[-100:]
                    else:
                        node.final_value=0
                        f_solution_list.append(node)
                if len(t_solution_list)<1:
                    continue
                t_solution_list=self.chose_solution(t_solution_list,2)
                f_solution_list=self.chose_solution(f_solution_list,8-len(t_solution_list))
                for chose_node in t_solution_list+f_solution_list:
                    save_data["solutions"].append({"solution":chose_node.cur_step,"generator":generator_name,"label":chose_node.final_value,"extract_answer":extract_answer(chose_node.cur_step)})
                save_data[f"{generator_name}_solved"]=solved
                save_data[f"{generator_name}_sample_num"]=sample_num
                with open(self.save_dir,"a") as file:
                    file.write(json.dumps(save_data) + '\n')
        
    def generate_train_solution_fixed(self):
        random.seed(self.seed)
        history = []

        llm = LLM(
            model=self.model_dir,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            seed=69,
        )
        generator_name = get_model_name(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        data_list = []
        # Just read all lines, and only require "problem" and "ground_truth_answer"
        with open(self.data_dir, "r") as f:
            data_list = json.load(f)

        random.shuffle(data_list)

        for indx, data in tqdm(enumerate(data_list), total=len(data_list), desc="Main process"):
            if self.start <= indx < self.end:
                problem = data["problem"]
                answer = data["ground_truth_answer"]

                t_solution_list, f_solution_list = [], []
                question_node = ag_step(question=problem, ans=answer)
                # print(question_node.question)
                save_data = {
                    "question": question_node.question,
                    "ground_truth": question_node.ans,
                    "solutions_uncertainty_driven": [],
                    "solutions_similarity_driven": []
                }

                solved = 0
                generate_prompt = get_prompt(question_node, history) if len(history) > 1 else None
                sample_num = 32  # Or any number you like

                solution_nodes = generate_numi_steps_test(
                    question_node, tokenizer, llm, sample_num, -10,
                    temperature=0.8, top_p=1.0,
                    ori_prompt=generate_prompt, max_tokens=2048
                )
                # print(solution_nodes)
                for node in solution_nodes:
                    # print(node.cur_step)
                    # print(node.step_token_probs)
                    # print(len(node.step_token_probs)==node.num_step_tokens[1])
                    generate_answer = extract_answer(node.cur_step)
                    if generate_answer == "" or generate_answer == "bad_answer":
                        continue
                    if math_is_equiv(generate_answer, question_node.ans):
                        node.final_value = 1
                        solved += 1
                        history.append({
                            "instruction": f"Instruction:{node.question}\nResponse: Let’s think step by step.\n{node.previous_steps}",
                            "response": node.cur_step
                        })
                        t_solution_list.append(node)
                        history = history[-100:]
                    else:
                        node.final_value = 0
                        f_solution_list.append(node)
                if len(t_solution_list) < 1:
                    continue
                # Sample: take 2 positive, and up to 6 negative solutions
                t_solution_list_chosen = self.choose_solution_uncertainty_driven(t_solution_list, 2, alpha=1.0)
                f_solution_list_chosen = self.choose_solution_uncertainty_driven(f_solution_list, 8 - len(t_solution_list_chosen), alpha=1.0)
                t_solution_list_chosen2 = self.chose_solution(t_solution_list, 2)
                f_solution_list_chosen2 = self.chose_solution(f_solution_list, 8 - len(t_solution_list_chosen))
                print("t_solution_list_original==t_solution_list_u_driven", t_solution_list_chosen==t_solution_list_chosen2)
                print("f_solution_list_original==f_solution_list_u_driven", f_solution_list_chosen==f_solution_list_chosen2)
                for chose_node in t_solution_list_chosen+f_solution_list_chosen:
                    save_data["solutions_uncertainty_driven"].append({
                        "solution": chose_node.cur_step,
                        "generator": generator_name,
                        "step_token_logprobs": chose_node.step_token_probs,
                        "label": chose_node.final_value,
                        "extract_answer": extract_answer(chose_node.cur_step)
                    })
                for chose_node in t_solution_list_chosen2+f_solution_list_chosen2:
                    save_data["solutions_similarity_driven"].append({
                        "solution": chose_node.cur_step,
                        "generator": generator_name,
                        "step_token_logprobs": chose_node.step_token_probs,
                        "label": chose_node.final_value,
                        "extract_answer": extract_answer(chose_node.cur_step)
                    })
                save_data[f"{generator_name}_solved"] = solved
                save_data[f"{generator_name}_sample_num"] = sample_num
                with open(self.save_dir, "a") as file:
                    file.write(json.dumps(save_data) + '\n')

                # with open("log_t.txt", "a") as file:
                #     file.write(str(t_solution_list_chosen) + '\n')
                # with open("log_f.txt", "a") as file:
                #     file.write(str(f_solution_list_chosen) + '\n')

if __name__=="__main__":
    args = parse_args()
    labeler=generate_prm_data(args)
    if args.generate:
        print("=========================Generating solution===========================")
        labeler.generate_train_solution_fixed()
    elif args.get_label:
        print(f"=========================Getting label================================")
        labeler.get_label()