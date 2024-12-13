import requests
import json
import pandas as pd
import csv
import copy
from openai import OpenAI

model_list = ['o1-preview-2024-09-12','o1-mini-2024-09-12']
model_name = 'o1-mini-2024-09-12'
tgt_file = "results_score_o1-mini-2024-09-12/res_version1_sys_his_sub1.csv"
#tgt_file = "results_score_o1-mini-2024-09-12/res_tmp.csv"
df = pd.read_csv('./stim_LLM/version1_sys.csv')
#df = pd.read_csv('./stim_LLM/v1_tmp_sys.csv')
df_new = pd.DataFrame()
choice_dict = {'A':'inf', 'B':'social', 'C':'both'}

system_prompt = "想象以下双人对话场景，你在其中扮演“你”这一角色，请对方对某事/物做出评价。识别特定人物的话语中的真实意图，在给出的三个选项中选择一个你认为的正确答案。假设有以下场景:\
你烤了一个蛋糕，\
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

from openai import OpenAI
client = OpenAI(
    api_key="",
    base_url="https://api.openai.com/v1/",
)

# 初始化对话历史
conversation_history = [ 
              #{"role": "system", 
              #"content": system_prompt},
              {"role": "user", 
              "content": system_prompt},
              {"role": "assistant",
              "content": prompt_a},
              ]
#conversation_history = copy.deepcopy(INIT_HIS)

with open(tgt_file, mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"])

for i, row in df.iterrows():
  context = row['Dialogue']
  question = row['Question']
  content_text = context+'\n'+question
  
  # 将模型的回复添加到上下文中
  if i >0:
    conversation_history.append({"role": "assistant", "content": answer})
  if len(conversation_history) == 146: ##72*2+2， 新block
    conversation_history = [ 
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt_q},
              {"role": "assistant","content": prompt_a},
              ]
    print('############## conversation_history ',len(conversation_history),conversation_history)
  # 添加新的问题
  conversation_history.append({"role": "user", "content": content_text})
  '''
  Other: temperature and top_p are fixed at 1, while presence_penalty and frequency_penalty are fixed at 0.
  '''
  #print(model_name)
  completion = client.chat.completions.create(
    model=model_name,
    messages=conversation_history,
    seed=1)
      #temperature=0.7,
      #presence_penalty=1.1,
      #max_tokens=512,
      #top_p=0.9)
  
  r = json.loads(completion.model_dump_json())

  answer = r['choices'][0]['message']['content'].strip().replace('\n','')
  #print('<<<<<<< 1 ans', answer)
  position = answer.index("我的选择是：")
  #print(f"字符'：'的位置是：{position}")
  choice_item = answer[position+6:position+7]
  if choice_item in choice_dict.keys():
    choice = choice_item
  else:
    if 'A.' in answer:
      print('<<< A OK')
      choice = 'A'
    elif 'B.' in answer:
      print('<<< B OK')
      choice = 'B'
    elif 'C.' in answer:
      print('<<< C OK')
      choice = 'C'
  
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
    
  with open(tgt_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(df.loc[i, ["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"]])
  print("{} trials complete!".format(i+1))
