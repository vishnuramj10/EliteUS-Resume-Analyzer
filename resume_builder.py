# -*- coding: utf-8 -*-
import fitz
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd
import numpy as np
import json
import pandas as pd
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from transformers import AutoTokenizer, TextStreamer, pipeline,LlamaForCausalLM,AutoModelForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from accelerate import Accelerator
from langchain.chains import LLMChain

def load_resume(resume_file):

  doc = fitz.open(resume_file)
  resume_text = ""
  for page in doc:
      text = page.get_text()
      output = page.get_text("blocks")
      output
      for block in output:
          resume_text += block[4]
  return resume_text

def load_model(model_name):

  hf_token = "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"
  os.environ['HF_TOKEN'] = hf_token
  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      trust_remote_code=True,
      torch_dtype=torch.float16,
      # load_in_8bit=True,
      # load_in_4bit=True,
      device_map="auto",
      use_cache=True,
  )
  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

  text_pipeline = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=5000,
      do_sample=False,
      repetition_penalty=1.15,
      streamer=streamer
  )
  llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.1})
  accelerator = Accelerator()

  config = {'max_new_tokens': 512, 'repetition_penalty': 1.1, 'context_length': 8000, 'temperature':0, 'gpu_layers':50}
  llm, config = accelerator.prepare(llm, config)
  
  return llm

def get_skills(llm, resume_text):

  skill_prompt = """From the Resume text below, extract Entities strictly as instructed below:

  1. Look for Experience Entities in the text.
      Entity Definition:
      label:'Skills',name:string,category:string,years:string //Skills Node

  2. Get Top 10 technical skills from each professional experience.

  3. Do NOT create duplicate entities.

  4. Have just one technical skills section for each professional experience.

  5. NEVER Impute missing values.

  6. Calculate duration as end_date - start_date of professional experience and express it in a decimal format. For example, 1/May 2021 - June 2022 = 1.1, 2/May 2021 - June 2021 = 0.1.

  7/ Follow the JSON format strictly.

  8/ Give all the companies the person has worked at beginning and the skills used.

  Output JSON as an example (Follow this JSON format strictly):
  {{"entities":{{"Amazon":[{{ "skills": ["Java", "SQL", "c","d","e","f","g","i","h","k"]}}, {{"duration":"2.9"}}]}}}}

  Question: Now, extract entities as mentioned above for the text below -

  {resume_text}

  Answer:
  """

  prompt_template = PromptTemplate(template=skill_prompt,input_variables=['resume_text'])
  chain = LLMChain(llm=llm, prompt=prompt_template)
  result = chain(resume_text)

  start_index = result['resume_text'].index('Answer')
  sliced_text = result['resume_text'][start_index:]

  sliced_text = sliced_text.replace('Answer:\n', '')
  json_data = json.loads(sliced_text)
  extracted_skills = [json_data['entities']]

  return extracted_skills

def fetch_required_skills(required_experience):

  required_skills_list = required_experience.split(", ")
  required_skills = []
  required_years = []

  # Iterate through the values list and extract skill and experience
  for skill in required_skills_list:
      req_skill, experience = skill.split(" ")
      required_skills.append(req_skill.title())  # Ensure title case for skill
      required_experiences.append(int(experience))

  # Create a DataFrame
  df_required_skills = pd.DataFrame({
      'skill': required_skills,
      'required_experience': required_experiences
  })

  return df_required_skills

def calculate_skill_experience(extracted_skills):

  company_list = []
  skills_list = []
  duration_list = []
  for company_data in extracted_skills:
      for company_name, skills_data in company_data.items():
          for skills_info in skills_data:
              skills = skills_info['skills']
              if 'duration' in skills_info:
                duration = skills_info['duration']
              skills_list.extend(skills)
              duration_list.extend([duration] * len(skills))
              company_list.extend([company_name] * len(skills))

  df = pd.DataFrame({'company': company_list, 'skill': skills_list, 'duration': duration_list})
  df['duration'] = df['duration'].astype('float')
  df_grouped = df.groupby('skill', as_index=False)['duration'].sum()
  df_grouped.rename(columns={'duration':'experience'}, inplace=True)
  df_grouped['skill'] = df_grouped['skill'].str.title()
  return df_grouped

def get_skills_json(df_required_skill, df_grouped):

  df_skills = df_required_skill.merge(df_grouped, on='skill', how='left')
  df_skills['experience'].fillna(0, inplace=True)
  json_data = []
  for index, row in df_skills.iterrows():
    skill_info = {
          row['skill']: {
              'required_experience': row['required_experience'],
              'experience': row['experience']
          }
    }
    json_data.append(skill_info)
  return json_data