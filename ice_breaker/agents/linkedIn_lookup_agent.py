from dotenv import load_dotenv
import os

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Function to find the LinkedIn profile URL of a person given their full name
from tools.tools import get_linkedin_profile_url_tavily


def lookup(name: str) -> str:
    # Initialize the language model (LLM) with zero temperature for deterministic output
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Define a prompt template that asks the LLM to return only the LinkedIn URL
    template = """given the full name {name_of_person} I want you to get me a link to their linkedIn profile page.
    Your answer should only contain the URL"""

    # Wrap the raw string template into a LangChain PromptTemplate object
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # Define a tool that can be used by the agent (stubbed for now, must implement crawling logic)
    tools_for_agent = [
        Tool(
            name="crawl google for linkedIn profile page",
            func=get_linkedin_profile_url_tavily(name=name),
            description="useful for when you need to get the LinkedIn page URL"
        )
    ]

    # Load a pre-built ReAct agent prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # Create a ReAct-style agent using the specified LLM, tools, and prompt
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # Create an executor to run the agent with verbose logging enabled
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Format the prompt using the input name and invoke the agent
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    # Extract the LinkedIn profile URL from the agent's output and return
    linkedin_profile_url = result["output"]
    return linkedin_profile_url


if __name__ == "__main__":
    load_dotenv()
    print("Hey " + os.environ['DEVELOPERS_NAME'] + ", How u doing?")
    linkedin_url = lookup(name="Sahil kumar")
    print(f"FINAL OUTPUT: {linkedin_url}")
