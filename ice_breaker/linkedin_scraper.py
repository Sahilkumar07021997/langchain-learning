from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello Langchain")
    print("Hey " + os.environ['DEVELOPERS_NAME'] + ", How u doing?")

    # Summary template
    summary_template = """given the LinkedIn profile {data} about person i want to you to create:
    1. A short summary
    2. Two interesting facts
    3. Achievements
    4. Better job role fit
    """

    # Summary prompt template
    summary_prompt_template = PromptTemplate(input_variables=["data"], template=summary_template)

    # Llm chat model - ChatOpenAI and llama3.1(opensource model running locally)
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm = ChatOllama(model="llama3.1:latest")

    # Chaining
    chain = summary_prompt_template | llm | StrOutputParser()

    # LinkedIn Data
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/sahil-kumar-aa868218b",
                                            mock=True)

    # Response
    res = chain.invoke(input={"data": linkedin_data})
    print(res)
