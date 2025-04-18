import sqlite3
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

from src.dpe_optimization.tools.custom_tool import DatabaseQueryTool


@CrewBase
class AnalystCrew():
    """Analyst Crew"""

    def __init__(self, custom_inputs: dict):
        super().__init__()

        self.custom_inputs = custom_inputs

        # LLM Prompt
        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"
        
        # LLM Config
        self.custom_llm = LLM(
            model="openai/gpt-4o-mini", # call model by provider/model_name
            temperature=0, # 0.1-0.3 for factual response (best practice)
            timeout=300, # Longer timeout for complex tasks
            # seed=42,
        )

        # Knowledge Source: Column Name
        # content = "A list of column name is ['Date', 'Category', 'Amount', 'Note'] and table name is 'records'"
        content = "A list of column name is ['Date', 'Category', 'Amount', 'Note'], table name is 'records', and local database name is 'test.db'"
        self.string_source = StringKnowledgeSource(
            content=content,
        )

        # Tools
        self.query_database = DatabaseQueryTool()

    # @tool
    # def query_database(query: str):
    #     """Execute SQL query and return the result"""
    #     conn = sqlite3.connect("test.db")
    #     cursor = conn.cursor()
    #     cursor.execute(query)
    #     results = cursor.fetchall()
    #     conn.close()
    #     return results

    @agent
    def analyze_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["analyze_agent"],
            llm=self.custom_llm,
            tools=[self.query_database],
            max_iter=3,
            max_rpm=10
        )
    
    @task
    def analyze(self) -> Task:
        return Task(
            config=self.tasks_config["analyze"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[self.string_source],
            memory=False, # stateless execution
            verbose=True,
        )
