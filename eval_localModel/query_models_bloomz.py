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
torch.mps.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)

# Prompt embedding
#DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
#DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant, you are good at analyzing interlocutor's mental state. 你是一个乐于助人的助手，你擅长分析对方的心理状态。"""
DEFAULT_SYSTEM_PROMPT = "想象以下双人对话场景，你在其中扮演“你”这一角色，请对方对某事/物做出评价。识别特定人物的话语中的真实意图，在给出的三个选项中选择一个你认为的正确答案。假设有以下场景:"
prompt_q = "你烤了一个蛋糕，\
如果有五个等级，用五点量表表示，数字越小对应评价越差（等级越低），数字越大表示评价越优（等级越高）。\
你对蛋糕的评估大概率（约80%）处于等级“1”和“2”的范围。\
给小琳品尝后，你问道：“我烤的蛋糕怎么样？”\
小琳回答道：“不是逊色的。”\
请你根据以上情景判断小琳说\"不是逊色的。\"的真实意图是：\
A.仅真实评价（对方仅仅表达对事/物优劣的真实评价，而不考虑你的面子或感受）\
B.仅维护面子（对方的表达仅仅考虑维护你的面子或感受，而不考虑评价的真实性）\
C.两者都兼顾（对方的表达同时考虑维护你的面子、以及评价的真实性）\
你的选择是："
prompt_a = "我的选择是：C.两者都兼顾（对方的表达同时考虑维护你的面子、以及评价的真实性）。"


choice_dict = {'A':'inf', 'B':'social', 'C':'both', 'X':'temp'}

general_template = ("{system_prompt}{instruction}")
def generate_prompt(instruction, template, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return template.format_map({'instruction': instruction,'system_prompt': system_prompt})
def generate_message(user_message, system_prompt=DEFAULT_SYSTEM_PROMPT):
    context = f"{system_prompt}\nUser: {user_message}\nAssistant:"
    return context

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, \
T5ForConditionalGeneration, LlamaTokenizer
from transformers import GenerationConfig

def load(model_name="bloomz-7b1", model_id="bigscience/bloomz-7b1", cache_dir='/Users/xiaoye/Downloads/local_LLM/'):
    model_dir = cache_dir+'local_'+model_name
    if "alpaca" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
        print(f"Successfully loaded model ({model_name})")
        tokenizer =  LlamaTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,trust_remote_code=True)
        #model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto',trust_remote_code=True,device_map='auto',offload_folder="offload",offload_state_dict=True)
        print(f"Successfully loaded model ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("{} is available.".format(device))
    model.to(device).eval()
    return model, tokenizer

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def get_generation(model, model_id, tokenizer, message, generation_config, **kwargs):
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--model_name', type=str, default="gpt-4")
    argparser.add_argument('-m', '--model_id', type=str, default="google/flan-t5-base", help="model id from huggingface.co/models or model path (gguf) from local disk.")
    argparser.add_argument('--exp_dir', type=str, default="exp")
    argparser.add_argument('--index', action="store_true", default=False, help="Whether to name the response files with indices of questions in the original form.")

    args = argparser.parse_args()

    #df = pd.read_csv('../stim_LLM_modFiller/version1_sys.csv')
    df = pd.read_csv('../stim_LLM/v1_tmp_sys.csv')
    model_name = args.model_name
    model_id = args.model_id
    exp_dir = args.exp_dir

    # Check if exp_dir exists
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    csv_file = f"res_version1_sys_{model_name}_sub1.csv"
    #csv_file = f"res_tmp_sys_{model_name}.csv"
    
    with open(f"{exp_dir}/{csv_file}", mode='w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(["Index", "prompt_main", "choices", "choice_annotation", 'num_choices', 'compOrChat', 'choices_aft_rand', 'correct_aft_rand', 'temperature', 'top_p', 'seed', "generation", "generation_isvalid", "distribution", "prob_true_answer", "model_answer",  "correct", "model_answer_condition"])
        writer.writerow(["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"])
    
    #model, tokenizer = load(model_id)
    model, tokenizer = load(model_name, model_id)
    conversation_history = {}
    df_new = pd.DataFrame()
    for i, row in tqdm(df.iterrows()):
        context = row['Dialogue']
        question = row['Question']
        message_main = context+'\n'+question
        '''
        message = [ 
              {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
              #{"role": "user", "content": prompt_q},
              #{"role": "assistant", "content": prompt_a},
              {"role": "user", "content": message_main},
              ]
        '''
        message = f"{DEFAULT_SYSTEM_PROMPT}\nUser: {message_main}\nAssistant:"
        print('################## input \n', message)
        generation_config = {
            "max_new_tokens": 1,
            "temperature": 0.7,
            "top_p":0.9, 
            "repetition_penalty":1.1,
            "do_sample": True, 
            #"top_k": 1,
        }
        
        answer = get_generation(
            model, model_id, tokenizer, message, generation_config
        ).strip().replace('\n','')
        print('<<<<<<< 1 ans', answer)
        if "：" in answer:
            position = answer.index("：")
            #print(f"字符'：'的位置是：{position}")
            choice_item = answer[position+1:position+2]
            if choice_item in choice_dict.keys():
                choice = choice_item
        elif 'A' in answer:
            print('<<< A OK')
            choice = 'A'
        elif 'B' in answer:
            print('<<< B OK')
            choice = 'B'
        elif 'C' in answer:
            print('<<< C OK')
            choice = 'C'
        else:
            choice = 'X'
        
        print('<<<<<<< 2 ans ', choice_dict[choice])

        df.loc[i, "answer"] = answer
        df.loc[i, "choose_goal"] = choice_dict[choice]
        df.loc[i, "score"] = (choice_dict[choice] in row['True_Goal'])

        df.loc[i, "model"] = model_name
        df.loc[i, "trial_type"] = row['trial_type']
        df.loc[i, "condition"] = row['condition']
        df.loc[i, "item"] = row['item']
        df.loc[i, "true_goal"] = row['True_Goal']
        df.loc[i, "model"] = model_name
        
        with open(f"{exp_dir}/{csv_file}", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(df.loc[i, ["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"]])
