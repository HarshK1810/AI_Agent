from langchain_core.messages import HumanMessage
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()  #to load the API Key

@tool
def calculator(a: float, b:float) -> str:
    """Useful for basic arithmetic operations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"

@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are doing good today"

# System prompt to control model behavior
system_prompt = """You are a helpful assistant with access to tools.
- Call a tool only ONCE per user request.
- After receiving a tool result, respond naturally to the user using that result
- Never repeat tool calls.
- Never print raw function call syntax in your response.
"""


def main():
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  #to create a model using ChatOpenAI 

    tools = [calculator, say_hello]
    agent_executor = create_react_agent(model, tools, prompt=system_prompt) # creating an agent executor to execute the agent with our given model and tools

    print("Welcome! I'm your AI Assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break
        
        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(                 
            {"messages":[HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:    
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
