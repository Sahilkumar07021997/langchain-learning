from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import render_text_description, Tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non-alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    # Register available tools
    tools = [get_text_length]

    # Prompt template that defines the ReAct format for the agent to follow
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # Fill in tool details into the prompt
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),  # Render all tool descriptions into text
        tool_names=", ".join([t.name for t in tools]),  # List tool names in prompt
    )

    # Initialize the OpenAI chat model with deterministic output (temperature=0)
    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation"],  # Stop generation after "Observation"
    )

    # List to store intermediate steps (thought/action/observation history)
    intermediate_steps = []

    # Compose the full agent pipeline using function piping (`|`)
    agent = (
            {
                "input": lambda x: x["input"],  # Extract input question
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),  # Format intermediate steps
            }
            | prompt  # Fill in the ReAct prompt
            | llm  # Send prompt to the LLM
            | ReActSingleInputOutputParser()  # Parse model output into AgentAction or AgentFinish
    )

    # First call to the agent with initial question and empty scratchpad
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    # If the agent returns an action (i.e., wants to use a tool)
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool  # Name of the tool to invoke
        tool_to_use = find_tool_by_name(tools, tool_name)  # Retrieve tool from list
        tool_input = agent_step.tool_input  # Extract tool input
        observation = tool_to_use.func(str(tool_input))  # Run the tool with input
        print(f"{observation=}")
        # Save the step result for next round of reasoning
        intermediate_steps.append((agent_step, str(observation)))

    # Invoke the agent again, now with one step already done
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)

    # If the agent is now ready with the final answer
    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)  # Display the final answer
