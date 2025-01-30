from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource


@CrewBase
class Poc():
	"""Poc crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	
	llm_model = LLM(
		model="gpt-4o",
		temperature=0.2
	)


	@agent
	def project_leader(self) -> Agent:
		return Agent(
			config=self.agents_config['project_leader'],
			allow_delegation=True,
			llm=self.llm_model,
			verbose=True
		)
	
	@agent
	def domain_leader(self) -> Agent:
		return Agent(
			config=self.agents_config['domain_leader'],
			llm=self.llm_model,
			allow_delegation=True,
			# knowledge_sources=[self.string_source], # knowledge_sources can be used in crew level or agent level
			verbose=True
		)
	
	@agent
	def data_engineer(self) -> Agent:
		return Agent(
			config=self.agents_config['data_engineer'],
			allow_delegation=True,
			llm=self.llm_model,
			verbose=True
		)


	@task
	def project_leader_task(self) -> Task:
		return Task(
			config=self.tasks_config['project_leader_task'],
			output_file='report.md'
		)
	
	@task
	def domain_leader_task(self) -> Task:
		return Task(
			config=self.tasks_config['domain_leader_task'],
		)
	
	@task
	def data_engineer_task(self) -> Task:
		return Task(
			config=self.tasks_config['data_engineer_task'],
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the Poc crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
			planning=True,
		)
