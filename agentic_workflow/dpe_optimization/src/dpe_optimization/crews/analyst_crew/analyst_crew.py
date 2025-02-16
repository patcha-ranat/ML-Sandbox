from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class AnalystCrew:
    """Analyst Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def analyze_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["analyze_agent"],
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
            verbose=True,
        )
