#!/usr/bin/env python
# coding: utf-8

# #### Notebook for running Chain-of-Thought with supporting context experiments 

# In[ ]:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()
n = args.n

import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-o5678901234567890'


# In[1]:


import sys, os
sys.path.append('..')
root = '../root/'


# In[2]:


import joblib
import numpy as np
from agents import CoTAgent, ReflexionStrategy
from util import summarize_trial, log_trial, save_agents


# #### Load the HotPotQA Sample

# In[4]:


hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

hotpot['supporting_paragraphs'] = None
for ind, row in hotpot.iterrows():
    supporting_articles = row['supporting_facts']['title']
    articles = row['context']['title']
    sentences = row['context']['sentences'] 
    supporting_paragraphs = []
    for article in supporting_articles:
        supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
        supporting_paragraphs.append(supporting_paragraph)
    supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
    hotpot.at[ind, 'supporting_paragraphs'] = supporting_paragraphs


# #### Define the Reflexion Strategy

# In[5]:


print(ReflexionStrategy.__doc__)


# In[6]:


strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION


# #### Initialize a CoTAgent for each question

# In[7]:


from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from fewshots import COT, COT_REFLECT
agents = [CoTAgent(row['question'],
                   row['supporting_paragraphs'],
                   row['answer'],
                   agent_prompt=cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt,
                   cot_examples=COT,
                   reflect_prompt=cot_reflect_prompt,
                   reflect_examples=COT_REFLECT,
                    ) for _, row in hotpot.iterrows()]


# #### Run `n` trials

# In[8]:



trial = 0
log = ''


# In[9]:


for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        agent.run(reflexion_strategy = strategy)
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_trial(agents, trial)
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

    with open(os.path.join(root, 'CoT', 'context', strategy.value, f'{len(agents)}_questions_{trial}_trials.result.txt'), 'w+') as f:
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}', file=f)



# #### Save the result log

# In[27]:


with open(os.path.join(root, 'CoT', 'context', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w+') as f:
    f.write(log)
# save_agents(agents, os.path.join(root, 'CoT', 'context', strategy.value, 'agents'))

