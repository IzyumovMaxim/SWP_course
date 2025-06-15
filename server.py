from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import re
import textstat
from groq import Groq

app = FastAPI()

# Initialize Groq client (replace with your actual API key)
client = Groq(api_key="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")

# Serve static files under /static; index.html served at root
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join("static", "index.html"))

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions for comment metrics

def compute_comment_density(code: str) -> float:
    lines = code.splitlines()
    code_lines = 0
    comment_lines = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if '/*' in stripped:
            in_block = True
        if in_block:
            comment_lines += 1
            if '*/' in stripped:
                in_block = False
            continue
        if stripped.startswith('//') or stripped.startswith('#'):
            comment_lines += 1
        else:
            code_lines += 1
    return (comment_lines / max(code_lines, 1)) * 100.0


def find_methods(code: str, language: str) -> list:
    patterns = {
        'java': r"\b(?:public|private|protected|static|final|synchronized|\s)+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{",
        'cpp': r"\b(?:void|int|float|double|char|bool|auto)\s+\w+\s*\([^)]*\)\s*\{"
    }
    pat = patterns.get(language.lower())
    return [m for m in re.finditer(pat, code)] if pat else []


def methods_with_comments(code: str, language: str) -> int:
    methods = find_methods(code, language)
    count = 0
    lines = code.splitlines()
    for m in methods:
        method_line_idx = code[:m.start()].count('\n')
        for i in range(1, 4):
            idx = method_line_idx - i
            if idx < 0:
                break
            prev = lines[idx].strip()
            if not prev:
                continue
            if prev.startswith('//') or prev.startswith('/*') or prev.endswith('*/'):
                count += 1
            break
    return count


def compute_methods_commented(code: str, language: str) -> float:
    methods = find_methods(code, language)
    total = len(methods)
    commented = methods_with_comments(code, language)
    return (commented / max(total, 1)) * 100.0


def compute_readability(code: str) -> float:
    comments = []
    for bc in re.findall(r'/\*([\s\S]*?)\*/', code):
        for line in bc.splitlines():
            stripped = line.strip()
            if stripped:
                comments.append(stripped)
    for lc in re.findall(r'//(.*)', code):
        stripped = lc.strip()
        if stripped:
            comments.append(stripped)
    if not comments:
        return 0.0
    scores = []
    for comment in comments:
        code_like_tokens = sum(1 for ch in comment if ch in '{}();=<>')
        if code_like_tokens > len(comment) * 0.1:
            continue
        text = comment
        if not re.search(r'[\.!?]\s*$', text):
            text = text + '.'
        try:
            score = textstat.flesch_reading_ease(text)
            scores.append(score)
        except Exception:
            continue
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def create_evaluation_prompt(filename, code):
    total_lines = len(code.splitlines())
    return f"""
You are a code-comment evaluator. Follow these rules exactly:

1. Compute these three metrics (as percentages):
   - comment_density = (comment_lines ÷ code_lines) × 100  
   - methods_commented = (methods_with_comments ÷ total_methods) × 100  
   - meaningless_comments = (meaningless_comments ÷ total_comments) × 100  

2. List every real missing/problematic comment in two columns:
   - line number (1–{total_lines})  
   - very short description (≤5 words)

Ensure output is valid CSV with no extra text. Example structure:
```
filename,comment_density,methods_commented,meaningless_comments
{filename},{{density:.1f}},{{methods:.1f}},{{meaningless:.1f}}
line,description
12,complex logic uncovered
``` 

**INPUTS**
- filename: the file name
- code: the full source text

**STUDENT CODE**
{code}
"""

class EvaluateRequest(BaseModel):
    filename: str
    content: str
    options: dict

@app.post("/api/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    filename = req.filename
    code = req.content
    ext = os.path.splitext(filename)[1].lower()
    lang = 'java' if ext == '.java' else 'cpp' if ext in ['.cpp','c','.hpp','.cxx','.cc'] else ''
    density = compute_comment_density(code)
    methods_pct = compute_methods_commented(code, lang)
    readability = compute_readability(code)
    prompt = create_evaluation_prompt(filename, code)
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role":"user","content": prompt}],
                temperature=0.3
            )
        )
        feedback = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {e}")
    meaningless_pct = 0.0
    issues = []
    for line in feedback.splitlines():
        if line.lower().startswith(f"{filename.lower()},"):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                try:
                    meaningless_pct = float(parts[3])
                except:
                    pass
        elif line and line[0].isdigit():
            parts = [p.strip() for p in line.split(',', 1)]
            if len(parts) == 2:
                issues.append(f"{parts[0]}:{parts[1]}")
    result_obj = {
        "filename": filename,
        "density": round(density, 1),
        "methods_pct": round(methods_pct, 1),
        "readability": round(readability, 1),
        "meaningless_pct": round(meaningless_pct, 1),
        "issues": issues
    }
    return [result_obj]

# To run: uvicorn server:app --reload --host 0.0.0.0 --port 8000
