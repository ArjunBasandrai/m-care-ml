from openai import OpenAI
from dotenv import load_dotenv

def init():
    load_dotenv()
    client = OpenAI()
    sys_prompt = {"role": "system", "content": "You are a helpful assistant for mothers during postpartum."}
    discussion = [sys_prompt]
    return client, sys_prompt, discussion

def get_input(discussion, sys_prompt):
    if len(discussion) > 1:
        user_input = input("You: ")
        if user_input == "exit":
            return None
        discussion.append({"role": "user", "content": user_input})
    if len(discussion) > 10 + 1:
        discussion = discussion[-10:]
        discussion.insert(0, sys_prompt)
    return discussion

def get_response(client, discussion):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=discussion,
        stream=True,
    )

    message = ""
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text is not None:
            message += text

    discussion.append({"role": "system", "content": message})
    
    return message, discussion

def chatbot():
    client, sys_prompt, discussion = init()
    while True:
        discussion = get_input(discussion, sys_prompt)
        if discussion is None:
            break
        message, discussion = get_response(client, discussion)
        print(f"Bot: {message}")
