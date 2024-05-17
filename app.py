import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from fastai.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import urllib.request
import json
from resume_builder import *

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])

# Load the model and setup the question-answering chain
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
llm = load_model(model_name)

#@app.route('/resume', method=['POST'])
@app.route('/resume')
async def analyze(request):
    #resume = await request.form()
    resume_text = load_resume("Dasari Rana.docx")
    extracted_skills = get_skills(llm, resume_text)
    df_grouped = calculate_skill_experience(extracted_skills)
    df_required_skills = fetch_required_skills("Python 2, Oracle 5, angular 9, java 10")
    skills_json = get_skills_json(df_required_skills, df_grouped)

    return JSONResponse({'result': str(skills_json)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5500, log_level="info")