"""Entry point for the prototype CLI.
Demonstrates:
- Conversational flow with memory
- Asking document QA questions
- Asking weather
- Generating an image
- Running SQL queries
- Asking for recommendations
"""
import os
from controller import Controller

def demo():
    print("Integrated Multi-Agent Prototype (CLI)")
    print("Type 'help' for commands. 'exit' to quit.\n")
    ctrl = Controller(memory_size=6)
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user.lower() == 'help':
            print("Commands/examples:\n - 'what's the weather in Tokyo?'\n - 'generate image: a red kite flying over mountains'\n - 'qa: what is agile project management?'\n - 'sql: select * from products where price < 50'\n - 'recommend: productivity books'\n")
            continue
        response = ctrl.handle(user)
        print("Assistant:", response)

if __name__ == '__main__':
    demo()
