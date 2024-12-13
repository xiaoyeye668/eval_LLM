import requests
import json
import pandas as pd
import csv

model_name = 'gpt-4o-2024-08-06'


tgt_file_list = [
    "results_score_gpt-4o/res_version2_sys_his_sub4.csv",
    "results_score_gpt-4o/res_version3_sys_his_sub5.csv",
]

choice_dict = {'A':'inf', 'B':'social', 'C':'both'}

system_prompt = "想象以下双人对话场景，你在其中扮演“你”这一角色，请对方对某事/物做出评价。识别特定人物的话语中的真实意图，在给出的三个选项中选择一个你认为的正确答案。假设有以下场景:"
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


for files in tgt_file_list:
  filename = files.split('/')[-1]
  version = filename[:-4].split('_')[1]
  subid = int(filename[:-4].split('_')[4][3:])
  print('<<<<<< ',version, subid)
  df = pd.read_csv('./stim_LLM/{}_sys.csv'.format(version))

  api_url = "https://api.openai.com/v1/chat/completions"
  headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer xxx"
  }

  with open(files, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"])

  for i, row in df.iterrows():
    context = row['Dialogue']
    question = row['Question']
    content_text = context+'\n'+question
    
    data = {
      "model": model_name,
      "messages":[ {"role": "system", "content": [{ "type": "text", "text":system_prompt}]},
                  {"role": "user", "content": [{ "type": "text", "text": prompt_q}]},
                  {"role": "assistant","content": [{ "type": "text", "text": prompt_a}]},
                  {"role": "user", "content": content_text}],
      "temperature": 0.7,
      "presence_penalty": 1.1,
      "max_tokens": 512,
      "seed": subid,
      "top_p":0.9
    }
    
    response = requests.post(api_url, headers=headers, json=data)
    r = response.json()
    
    print(r)
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
      
    with open(files, mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(df.loc[i, ["model", "trial_type", "condition", "item", 'true_goal', "answer", "choose_goal", "score"]])
    print("{} trials complete!".format(i+1))
