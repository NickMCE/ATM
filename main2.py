import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Load LLM models (replace with your preferred methods)
api_gemini = "AIzaSyAYXlh8OBQyWtvXcQsvVQzUYLOgI80NbDM"
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, google_api_key=api_gemini
)

# Define agents with roles and LLMs
marketer = Agent(
    role="Market Research Analyst",
    goal="Find out market demand...",
    backstory="You are a market research analyst with 2 decades of experience analyzing market valuation, stock prices, etc.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

technologist = Agent(
    role="Technology Expert",
    goal="Assess technological feasibility...",
    backstory="You are a top techie with experience in both present and emerging technologies. You have worked for top tech companies like Apple, Google, and Meta.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

business_consultant = Agent(
    role="Business Development Consultant",
    goal="Evaluate and advise on business model...",
    backstory="You are a successful business magnate who has helped expand businesses from national to international levels.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

agents = {'Marketer': marketer, 'Techie': technologist, 'Business': business_consultant}

def run_task(description, expected_output, selected_agent):
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=selected_agent
    )
    
    try:
        crew = Crew(tasks=[task], agents=[selected_agent], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        return result
    except Exception as e:
        raise e

def main():
    st.title("Agent Task Manager")

    description = st.text_input("Enter the task description:")
    expected_output = st.text_input("Enter the output you expect:")

    agent_options = list(agents.keys())
    selected_agent_name = st.selectbox("Select an agent for the task:", agent_options)

    if st.button("Create and Run Task"):
        selected_agent = agents.get(selected_agent_name)
        if not selected_agent:
            st.error(f"Selected agent '{selected_agent_name}' is not available.")
            return

        # Run the task synchronously
        try:
            result = run_task(description, expected_output, selected_agent)
            if result:
                st.success("Task completed!")
                
                # Display the result
                st.write(result)
            else:
                st.warning("No execution flow captured. Check the task and LLM output.")
        except Exception as e:
            st.error(f"Error running task: {e}")

if __name__ == "__main__":
    main()
