import numpy as np
import pandas as pd
import argparse
import os
import torch, glob
import sys
import csv
from tqdm import tqdm
torch.cuda.empty_cache()

SEED = 12
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Prompt embedding
#DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
#DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant, you are good at analyzing interlocutor's mental state. 你是一个乐于助人的助手，你擅长分析对方的心理状态。"""
DEFAULT_SYSTEM_PROMPT = "想象以下双人对话场景，你在其中扮演“你”这一角色，请对方对某事/物做出评价。识别特定人物的话语中的真实意图，在给出的三个选项中选择一个你认为的正确答案。假设有以下场景:"
choice_dict = {'A':'inf', 'B':'social', 'C':'both', 'X':'None'}

# Templates
Alpaca_template = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

general_template = ("{system_prompt}{instruction}")

def generate_prompt(instruction, template, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return template.format_map({'instruction': instruction,'system_prompt': system_prompt})
def generate_message(user_message, system_prompt=DEFAULT_SYSTEM_PROMPT):
    context = f"{system_prompt}\nUser: {user_message}\nAssistant:"
    return context

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, \
T5ForConditionalGeneration, LlamaTokenizer
from transformers import GenerationConfig

def load(model_name="bloomz-7b1", model_id="bigscience/bloomz-7b1", cache_dir='/root/'):
    model_dir = cache_dir+'local_'+model_name
    if "alpaca" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
        print(f"Successfully loaded model ({model_name})")
        tokenizer =  LlamaTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    
    elif 'glm' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,trust_remote_code=True)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto',trust_remote_code=True,device_map='auto',offload_folder="offload",offload_state_dict=True)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("{} is available.".format(device))
    model.to(device)
    return model, tokenizer

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def get_completion(prompt_main, model, model_id, tokenizer, generation_config, answer_choices=["A", "B", "C"], **kwargs):
    if "alpaca" in model_id: prompt = generate_prompt(prompt_main, Alpaca_template)
    else: prompt = generate_message(prompt_main)
    #print('##################\n', prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    answer_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) \
                        for i in answer_choices]
    
    # print(answer_token_ids)
    # test_prompt = "Which character is ranked at the first in the numeric order? 1, 2 or 3?"
    #max_input_length = model.config.max_position_embeddings  # 获取模型的最大输入长度
    #max_input_length = tokenizer.model_max_length
    #print(f"Max input length: {max_input_length}")
    print(f"Max input length: {input_ids['input_ids'].shape}")
    outputs = model.generate(
        **input_ids, 
        max_new_tokens=1,
        output_scores=True,
        num_return_sequences=1,
        return_dict_in_generate=True,
        #temperature=0.7, 
        #top_p=0.9,
        #do_sample=True,
        #repetition_penalty=1.1
        #generation_config = generation_config
    )
    #outputs = model.generate(input_ids, num_return_sequences=1, temperature=0.7, top_p=0.9)
    rets = tokenizer.batch_decode(outputs['sequences'])
    #output = rets[0].strip().replace(text, "").replace('</s>', "")
    print("Firefly：{}".format(rets), len(rets[0]))
    # if isinstance(outputs.scores, tuple):
    #     logits = outputs.scores[0][0]
    # else:
    #     logits = outputs.scores
    index = rets[0].index('Assistant:')
    print(index, rets[0][-1])
    logits = outputs.scores[0][-1]
    #print('<<<<<<<<<< logits ', logits)
    print(logits.shape)
    # print(logits[36])
    # print(logits[209])

    # openbuddy, chinese-alpaca 需要索引[1]，其他模型直接logits[answer_id].item()
    # if "openbuddy" in model_id.lower():
    #     answer_logits = [logits[answer_id][1].item() for answer_id in answer_token_ids]
    # else:
    answer_logits = [logits[answer_id].item() for answer_id in answer_token_ids] 
    #generated_answer = str(answer_choices[np.argmax(answer_logits)])
    #probs = softmax(answer_logits)
    #probs = {answer: probs[i] for i, answer in enumerate(answer_choices)}
    probs = {'A':0,'B':0,'C':0}
    generated_answer = rets[0][-1]
    return generated_answer, probs

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--style', choices=['verbose', 'concise'], default="concise", help="choose verbose to have responses kept in txt files.")
    argparser.add_argument('-n', '--model_name', type=str, default="gpt-4")
    argparser.add_argument('-m', '--model_id', type=str, default="google/flan-t5-base", help="model id from huggingface.co/models or model path (gguf) from local disk.")
    #argparser.add_argument('--cpp', action="store_true", default=False, help="Whether to use the CPP model.")
    argparser.add_argument('--exp_dir', type=str, default="exp")
    argparser.add_argument('--index', action="store_true", default=False, help="Whether to name the response files with indices of questions in the original form.")

    args = argparser.parse_args()

    df = pd.read_csv('./stim_LLM/version6_sys.csv')
    #df = pd.read_csv('./stim_LLM/v1_tmp_sys.csv')
    model_name = args.model_name
    model_id = args.model_id
    exp_dir = args.exp_dir

    # Check if exp_dir exists
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    csv_file = f"res_version6_sys_{model_name}_sub12.csv"
    with open(f"{exp_dir}/{csv_file}", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"])

    #model, tokenizer = load(model_id)
    model, tokenizer = load(model_name, model_id)
    conversation_history = {}
    for i, row in tqdm(df.iterrows()):
        context = row['Dialogue']
        question = row['Question']
        prompt_main = context+'\n'+question
        '''
        conversation_history[i] = prompt_main
        if i > 40:
            prompt_tmp = '\n'.join([conversation_history[i] for i in range(i-40, i)])
        else:
            prompt_tmp = '\n'.join([conversation_history[i] for i in range(i)])
        prompt_main = '\n'.join([prompt_tmp, prompt_main])
        #conversation_history = prompt_main
        '''
        answer_choices = ["A", "B", "C"]
        #max_new_tokens = 1 if row.compOrChat == 'comp' else 512
        max_new_tokens = 1
        # Deprecated, if generation_config is used, uncomment the line in the get_completion function. 
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9, 
            do_sample=True, 
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            seed=5
            )

        generated_answer, probs = get_completion(
            prompt_main, model, model_id, tokenizer, generation_config, answer_choices
        )
        if generated_answer in ['A','B','C']:
            generated_answer = generated_answer
        else:
            generated_answer = 'X'
        print('##### generated_answer, probs ', generated_answer, probs)
        
        sorted_probs = [probs[answer] for answer in answer_choices]
        chosen_answer = 'X'
        max_prob = 0
        for ans, prob in probs.items():
            if prob > max_prob:
                chosen_answer = ans
                max_prob = prob

        # Evaluate generated text.

        df.loc[i, "generation"] = generated_answer.strip()
        df.loc[i, "generation_isvalid"] = (generated_answer.strip() in answer_choices)
        # Record probability distribution over valid answers.
        df.loc[i, "answer"] = str(probs)
        
        # Take model "answer" to be argmax of the distribution.
        chosen_answer = 'X'
        max_prob = 0
        for ans, prob in probs.items():
            if prob > max_prob:
                chosen_answer = ans
                max_prob = prob
        chosen_answer = generated_answer
        df.loc[i, "choose_goal"] = chosen_answer
        print(choice_dict[chosen_answer])
        df.loc[i, "score"] = (choice_dict[chosen_answer] in row['True_Goal'])

        df.loc[i, "model"] = 'bloomz-7b1'
        df.loc[i, "true_goal"] = row['True_Goal']
        with open(f"{exp_dir}/{csv_file}", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(df.loc[i, ["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"]])