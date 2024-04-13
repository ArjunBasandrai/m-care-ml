from openai import OpenAI
from dotenv import load_dotenv
from chatbot import init, get_input, get_response

client, sys_prompt, discussion = init()

while True:
    discussion = get_input(discussion, sys_prompt)
    if discussion is None:
        break
    message, discussion = get_response(client, discussion)
    print(f"Bot: {message}")
