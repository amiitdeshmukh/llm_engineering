import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
from pathlib import Path
from system_info import retrieve_system_info

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OUTPUT_FILE=Path(__file__).with_name("output.md")

OUTPUT_CODE=Path(__file__).with_name("main.cpp")
EXECUTABLE = Path(__file__).with_name("main")


openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing from .env")

openai= OpenAI()

OPENAI_MODEL = "gpt-6-luna"

system_info = retrieve_system_info()
print(system_info)

message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.

If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
```python
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
```
Please tell me exactly what I should use for the compile_command and run_command.

System information:
{system_info}
"""

response = openai.chat.completions.create(
    model=OPENAI_MODEL, 
    messages=[
        {"role":"user","content":message}
    ])

output=response.choices[0].message.content

OUTPUT_FILE.write_text(
    output,
    encoding="utf-8"
)

compile_command = [
    "clang++",
    "-std=c++17",
    "-O3",
    str(OUTPUT_CODE),
    "-o",
    str(EXECUTABLE),
]

run_command = [str(EXECUTABLE)]


system_prompt = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""

def user_prompt_for(python):
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called main.cpp and then compiled and executed; the compilation command is:
{compile_command}
Respond only with C++ code.
Python code to port:

```python
{python}
```
"""

def messages_for(python):
    return [
        {"role": "system", "content":system_prompt},
        {"role":"user", "content":user_prompt_for(python)}
    ]

def write_output(cpp):
    with OUTPUT_CODE.open("w", encoding="utf-8") as f:
        f.write(cpp)

def port(client, model, python):
    reasoning_eff= "low"
    response = client.chat.completions.create(model=model, messages= messages_for(python), reasoning_effort=reasoning_eff)
    reply = response.choices[0].message.content
    reply = reply.replace('```cpp','').replace('```','')
    write_output(reply)

pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

def run_python(code):
    globals = {"__builtins__": __builtins__}
    exec(code, globals)

run_python(pi)

port(openai, OPENAI_MODEL, pi)