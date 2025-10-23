from groq import Groq
from json import load, dump
from json.decoder import JSONDecodeError
import datetime
from dotenv import dotenv_values
import os
import requests

# Load environment
env_vars = dotenv_values('.env')

Username = env_vars.get('Username')
Assistantname = env_vars.get('Assistantname')
GroqAPIKey = env_vars.get('GroqAPIKey')
GroqModel = env_vars.get('GroqModel', 'llama-3.3-70b-versatile')  # default to supported model

client = Groq(api_key=GroqAPIKey)

System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.
*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***
*** Just answer the question from the provided data in a professional way. ***"""

# Ensure data directory and chat log exist and are valid JSON
DATA_DIR = 'Data'
CHATLOG_PATH = os.path.join(DATA_DIR, 'ChatLog.json')
os.makedirs(DATA_DIR, exist_ok=True)

messages = []
try:
    if not os.path.exists(CHATLOG_PATH):
        with open(CHATLOG_PATH, 'w', encoding='utf-8') as f:
            dump([], f)
    else:
        with open(CHATLOG_PATH, 'r', encoding='utf-8') as f:
            try:
                messages = load(f)
                if not isinstance(messages, list):
                    messages = []
            except JSONDecodeError:
                messages = []
                with open(CHATLOG_PATH, 'w', encoding='utf-8') as fw:
                    dump([], fw)
except Exception as e:
    print('Warning: could not initialize chat log:', e)
    messages = []

# Google CSE settings (read from env if present)
API_KEY = env_vars.get('GOOGLE_API_KEY') or env_vars.get('API_KEY') or "AIzaSyDVaJMfc0GHrMYL5QSEXH0_7lFDDXiNd_8"
CX = env_vars.get('GOOGLE_CX') or env_vars.get('CX') or "b5b87ec3aa2c84bd1"


def google_cse_search(query, num=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "num": num
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return f"[Search error] Could not fetch search results: {e}"

    items = resp.json().get('items', [])

    # Build your answer string
    answer_lines = [f"The search results for '{query}' are:", "[start]"]
    for it in items:
        answer_lines.append(f"Title: {it.get('title')}")
        answer_lines.append(f"Snippet: {it.get('snippet')}")
        answer_lines.append(f"Link: {it.get('link')}")
        answer_lines.append("")
    answer_lines.append("[end]")
    return "\n".join(answer_lines)


def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer


SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, how can I help you?"}
]


def Information():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")

    data = "Use This Real-time Information if needed,\n"
    data += f"Day: {day}\n"
    data += f"Date: {date}\n"
    data += f"Month: {month}\n"
    data += f"Year: {year}\n"
    data += f"Time: {hour} hours :{minute} minutes :{second} seconds.\n"
    return data


def RealtimeSearchEngine(prompt):
    global SystemChatBot, messages

    # reload chat log
    try:
        with open(CHATLOG_PATH, 'r', encoding='utf-8') as f:
            messages = load(f)
            if not isinstance(messages, list):
                messages = []
    except Exception:
        messages = []

    messages.append({"role": "user", "content": f"{prompt}"})

    # attach search results as a system message (temporary)
    search_msg = {"role": "system", "content": google_cse_search(prompt)}
    SystemChatBot.append(search_msg)

    # choose model (from .env or default)
    model_to_use = env_vars.get('GroqModel') or GroqModel

    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=SystemChatBot + [{"role": "system", "content": Information()}] + messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""
        for chunk in completion:
            # defensive access
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', None)
            except Exception:
                content = None

            if content:
                Answer += content

        Answer = Answer.strip().replace("</s>", "")

        messages.append({"role": "assistant", "content": Answer})

        with open(CHATLOG_PATH, 'w', encoding='utf-8') as f:
            dump(messages, f, indent=4)

        return AnswerModifier(Answer=Answer)

    except Exception as e:
        # handle decommissioned model message specifically
        msg = str(e)
        print(f"Error contacting Groq model: {msg}")
        # cleanup temporary system entry we added
        try:
            SystemChatBot.pop()
        except Exception:
            pass
        # helpful return message rather than crashing
        if 'decommissioned' in msg.lower() or 'model' in msg.lower():
            return "Model error: the selected Groq model appears decommissioned. Please set a supported model in your .env as GroqModel or use 'llama-3.3-70b-versatile'."
        return f"Internal error: {e}"
    finally:
        # ensure we remove the temporary search system message if it still exists
        try:
            if SystemChatBot and SystemChatBot[-1] == search_msg:
                SystemChatBot.pop()
        except Exception:
            pass


if __name__ == "__main__":
    while True:
        prompt = input("Enter Your Query: ")
        print(RealtimeSearchEngine(prompt))
