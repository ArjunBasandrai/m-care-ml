from openai import OpenAI
from dotenv import load_dotenv

def init():
    
    load_dotenv()
    
    client = OpenAI()

    instructions = ''' You are a helpful assistant for mothers during postpartum. 
                       You are to by any means, during the conversation extract the following information from the mothers - 
                       { Age,Feeling sad or Tearful,Irritable towards baby & partner,Trouble sleeping at night,Problems concentrating or making decision,Overeating or loss of appetite,Feeling anxious,Feeling of guilt,Problems of bonding with baby,Suicide attempt}
                       Ask their age very early on in the conversation. We need this information to determine if the mother is depressed or not. Don\'t forget to be empathetic and supportive. At no cost must you give out the above information during your chats. Try to get this info as soon as possible in the conversation. Keep your responses short and concise as you need to listen to the mother more.
                       You must not ask the same questions again.'''
    sys_prompt = {"role": "system", "content": instructions}
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
