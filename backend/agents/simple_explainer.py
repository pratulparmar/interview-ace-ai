#=====================Imports===================================

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from backend.config.settings import get_settings, setup_langsmith


settings = get_settings()
setup_langsmith()
print("\n" + "="*60)
print("SIMPLE LANGCHAIN AGENT - TECH TERM EXPLAINER")
print("="*60)


print("\n Connecting to OpenAI GPT-4...")
llm = ChatOpenAI(
    model = settings.openai_model,
    temperature = 0.7,
    openai_api_key = settings.openai_api_key
)

print("Connected to GPT-4")

print("\n Creating Prompt template .....")

template = """
You are a helpful technical instructor explaining concepts to beginners.

Explain the following technical term in simple language that a beginner can understand:
Term: {term}

Provide:
1. A simple one-sentence definition
2. A real-world analogy
3. A practical example

Keep it concise and beginner-friendly.
"""

prompt = PromptTemplate (
    input_variables = ["term"],
    template = template
)


print("Prompt template created")

print("\n Creating LangChain Chain ....")

chain = prompt | llm

print("Chain Created")


test_terms = [
    "n8n",
    "saas",
    "DSA"
]


for term in test_terms:
    print(f"\n{'='*50}")
    print(f"Explaining:{term}")
    print('='*60)

    result = chain.invoke({"term":term})

    print(f"\n{result.content}")
    print("\n"+"="*60)


