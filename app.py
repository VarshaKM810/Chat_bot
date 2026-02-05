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
    return "Offline mode: " + (text if text else "")

def chat():
    global client, token, offline_mode
    messages = [SystemMessage("You are a helpful assistant.")]
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

