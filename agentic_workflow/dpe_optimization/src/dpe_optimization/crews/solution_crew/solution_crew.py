from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class SolutionCrew:
    """Solution Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def solution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["solution_agent"],
        )
    
    @agent
    def governance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["governance_agent"],
        )

    @task
    def provide_solution(self) -> Task:
        return Task(
            config=self.tasks_config["provide_solution"],
        )

    @task
    def comply_policy(self) -> Task:
        return Task(
            config=self.tasks_config["comply_policy"],
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
