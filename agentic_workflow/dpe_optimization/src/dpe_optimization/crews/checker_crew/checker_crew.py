from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class CheckerCrew:
    """Checker Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def inspect_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["inspect_agent"],
        )

    @task
    def check_root_cause_solution(self) -> Task:
        return Task(
            config=self.tasks_config["check_root_cause_solution"],
        )

    @task
    def prioritize_solution(self) -> Task:
        return Task(
            config=self.tasks_config["prioritize_solution"],
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
