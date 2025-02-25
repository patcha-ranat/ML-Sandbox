from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource


@CrewBase
class AnalystCrew():
    """Analyst Crew"""

    def __init__(self, custom_inputs: dict):
        super().__init__()

        self.custom_inputs = custom_inputs
        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"
        self.csv_source = CSVKnowledgeSource(
            file_paths=[f"data_{self.custom_inputs.get('exec_date')}.csv"]
        )

    @agent
    def analyze_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["analyze_agent"],
            llm="gpt-4o-mini"
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
            knowledge_sources=[self.csv_source],
            verbose=True,
        )
