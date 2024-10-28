#!/usr/bin/env python
# coding: utf-8

# #### Notebook for running React experiments

# 

# In[1]:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--strategy', type=str, choices=['NONE', 'LAST_ATTEMPT', 'REFLEXION', 'LAST_ATTEMPT_AND_REFLEXION'], default='REFLEXION')
args = parser.parse_args()
n = args.n
strategy: ReflexionStrategy = ReflexionStrategy[args.strategy]

import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-o5678901234567890'


# In[2]:


import sys, os
sys.path.append('..')
root  = '../root/'


# In[3]:


import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy


# #### Load the HotpotQA Sample

# In[4]:


hotpot = joblib.load('../data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)


# #### Define the Reflexion Strategy

# In[5]:


print(ReflexionStrategy.__doc__)


# In[6]:





# #### Initialize a React Agent for each question

# In[7]:


agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent
agents = [agent_cls(row['question'], row['answer']) for _, row in hotpot.iterrows()]


# #### Run `n` trials

# In[8]:


trial = 0
log = ''


# In[9]:


for i in range(n):
    for agent in [a for a in agents if not a.is_correct()]:
        if strategy != ReflexionStrategy.NONE:
            agent.run(reflect_strategy = strategy)
        else:
            agent.run()
        print(f'Answer: {agent.key}')
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
    with open(os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.result.txt'), 'w+') as f:
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}', file=f)


# #### Save the result log

# In[11]:


with open(os.path.join(root, 'ReAct', strategy.value, f'{len(agents)}_questions_{trial}_trials.txt'), 'w+') as f:
    f.write(log)
# save_agents(agents, os.path.join('ReAct', strategy.value, 'agents'))


# In[12]:


log


# In[13]:


"""
Finished Trial 1, Correct: 33, Incorrect: 38, Halted: 29
"""


# In[14]:


agents


# In[ ]:




