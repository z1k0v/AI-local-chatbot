from langchain_ollama import OllamaLLM
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate


template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer: 
"""

model = OllamaLLM(model="moondream") # -> need to try with llama3 (needs download 4.7GB+)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Welcome to the AI ChatBot, Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        result = chain.invoke({"context": context, "question": user_input})
        print("Bot: ", result)
        context += f"\nUser: {user_input}\nAI: {result}"


if __name__  == "__main__":
    handle_conversation()


# ollama run moondream "What's in this image? /Users/user/Documents/projects/p04/catan01.png" # -> worked in terminal

# image = Image.open("catan01.png")
# enc_image = model.encode_image(image)
# print(model.answer_question(enc_image, "Describe this image."))



