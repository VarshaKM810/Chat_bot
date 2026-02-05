import os
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

endpoint = os.environ.get("AZURE_AI_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com")
model = os.environ.get("AI_MODEL", "deepseek/DeepSeek-V3-0324")
token = os.environ.get("GITHUB_TOKEN")
client = None
offline_mode = False

if not token or not token.startswith("github_pat_"):
    offline_mode = True

if token:
    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token), model=model)
else:
    offline_mode = True

def offline_reply(text):
    t = (text or "").strip().lower()
    if not t:
        return "Tell me what you need help with."
    if any(g in t for g in ["hi", "hello", "hey", "namaste", "hola"]):
        return "Hello! How can I help you today?"
    if "how are you" in t:
        return "I'm good and ready to help. What do you need?"
    if "your name" in t:
        return "I'm your local assistant."
    if "help" in t:
        return "Share your goal and I’ll guide you step by step."
    if "joke" in t:
        return "Why did the developer go broke? Because they used up all their cache."
    if "time" in t and "date" in t:
        import datetime
        now = datetime.datetime.now()
        return f"It’s {now.strftime('%Y-%m-%d %H:%M')}."
    return "I’m offline, but I can still explain, brainstorm, or plan with you."

def chat():
    global client, token, offline_mode
    messages = [SystemMessage("You are a helpful assistant. Answer directly and do not repeat the user's message.")]
    if not offline_mode and client:
        initial = "Hi! How can I help you today?"
        print("Assistant: " + initial)
        messages.append(AssistantMessage(initial))
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        messages.append(UserMessage(user_input))
        if offline_mode or not client:
            assistant_content = offline_reply(user_input)
            print("Assistant: " + assistant_content)
            messages.append(AssistantMessage(assistant_content))
            continue
        try:
            response = client.complete(stream=True, messages=messages)
            assistant_content = ""
            print("Assistant: ", end="", flush=True)
            for update in response:
                if update.choices and update.choices[0].delta:
                    chunk = update.choices[0].delta.content or ""
                    assistant_content += chunk
                    print(chunk, end="", flush=True)
            print()
            if assistant_content:
                messages.append(AssistantMessage(assistant_content))
        except HttpResponseError as e:
            offline_mode = True
            assistant_content = offline_reply(user_input)
            print("Assistant: " + assistant_content)
            messages.append(AssistantMessage(assistant_content))
    if client:
        client.close()

if __name__ == "__main__":
    chat()

