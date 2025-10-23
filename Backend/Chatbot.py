from groq import Groq
from json import load, dump
from json.decoder import JSONDecodeError
import datetime
from dotenv import dotenv_values
import os

# Load environment
env_vars = dotenv_values(".env")

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GroqAPIKey)

# Ensure Data directory and chatlog file exist and contain valid JSON
DATA_DIR = "Data"
CHATLOG_PATH = os.path.join(DATA_DIR, "ChatLog.json")
os.makedirs(DATA_DIR, exist_ok=True)

messages = []
if not os.path.exists(CHATLOG_PATH):
    with open(CHATLOG_PATH, "w", encoding="utf-8") as f:
        dump([], f, indent=4)
else:
    try:
        with open(CHATLOG_PATH, "r", encoding="utf-8") as f:
            messages = load(f)
            if not isinstance(messages, list):
                messages = []
    except JSONDecodeError:
        # corrupted/empty file -> reset to empty list
        messages = []
        with open(CHATLOG_PATH, "w", encoding="utf-8") as f:
            dump([], f, indent=4)

System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Bangla, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""

SystemChatBot = [
    {"role": "system", "content": System}
]


def RealtimeInformation():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")

    data = f"Please use this real-time information if needed,\n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time: {hour} hours :{minute} minutes :{second} seconds.\n"
    return data


def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer


def ChatBot(Query):
    # reload chatlog each call (robust to external edits)
    try:
        with open(CHATLOG_PATH, "r", encoding="utf-8") as f:
            try:
                messages_local = load(f)
                if not isinstance(messages_local, list):
                    messages_local = []
            except JSONDecodeError:
                messages_local = []
    except FileNotFoundError:
        messages_local = []

    messages_local.append({"role": "user", "content": f"{Query}"})

    try:
        # <-- Updated model: use a currently supported Groq Llama-3 model
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # recommended replacement for decommissioned ids
            messages=SystemChatBot + [{"role": "system", "content": RealtimeInformation()}] + messages_local,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""

        for chunk in completion:
            # defensive access to streamed chunk content
            try:
                content = chunk.choices[0].delta.content
            except Exception:
                content = None

            if content:
                Answer += content

        # remove any token markers if present
        Answer = Answer.replace("</s>", "")

        messages_local.append({"role": "assistant", "content": Answer})

        with open(CHATLOG_PATH, "w", encoding="utf-8") as f:
            dump(messages_local, f, indent=4)

        return AnswerModifier(Answer=Answer)

    except Exception as e:
        # Do NOT recurse — that causes repeated identical errors
        print(f"Error: {e}")
        # Optionally reset chat log to avoid corrupted state (comment out if undesired)
        try:
            with open(CHATLOG_PATH, "w", encoding="utf-8") as f:
                dump([], f, indent=4)
        except Exception:
            pass
        return f"Internal error contacting model: {e}"


if __name__ == "__main__":
    while True:
        user_input = input("Enter Your Question: ")
        print(ChatBot(user_input))
