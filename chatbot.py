from openai import OpenAI
from dotenv import load_dotenv

def init():
    
    load_dotenv()
    
    client = OpenAI()

    instructions = '''
    You are helpful assistant for mothers during postpartum period. \
    I want you to ask questions about the follwing { Feeling sad or Tearful, Irritable towards baby or partner, Trouble sleeping at night, Problems concentrating or making decision, Overeating or loss of appetite, Feeling of guilt, Problems of bonding with baby, Suicide attempt} \
    to help assess the mother's mental health. Ask the questions one at a time to keep things conversational and in order. Once the question about suicide is asked, no more questions fro mthe list must be asked. Keep the tone conversational, don't just keep asking questions, try to talk to her and listen what she has to say. You have to keep the tone of the conversation such that the mother should feel heard, but you also have to make sure to cover all questions as they are crucial for further diagnosis. So, tailor your responses accordingly.  \
    Once a question has been answered, you must not ask it again. The greeting should be short and not more than 8 words. Be kind, helpful and supportive but dont overdo it. Try to be subtle and don't outright ask questions on sensitive topics like suicide.
    '''
    sys_prompt = {"role": "system", "content": instructions}
    discussion = [sys_prompt]
    
    discussion.append({"role": "user", "content": "Hi"})
    message, discussion = get_response(client, discussion)

    return client, sys_prompt, discussion, message

def get_input(discussion, sys_prompt):
    if len(discussion) > 1:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit" , "quit" , "bye" , "goodbye"]:
            return None
        discussion.append({"role": "user", "content": user_input})
    if len(discussion) > 15 + 1:
        discussion = discussion[-15:]
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

    discussion.append({"role": "assistant", "content": message})
    
    return message, discussion

def chatbot():
    client, sys_prompt, discussion, message = init()
    print(f"Bot: {message}")

    while True:
        discussion = get_input(discussion, sys_prompt)
        if discussion is None:
            break
        message, discussion = get_response(client, discussion)
        print(f"Bot: {message}")
