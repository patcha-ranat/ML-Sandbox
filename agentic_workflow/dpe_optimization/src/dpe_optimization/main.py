import logging
from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from dpe_optimization.crews.analyst_crew.analyst_crew import AnalystCrew
from dpe_optimization.crews.solution_crew.solution_crew import SolutionCrew
from dpe_optimization.crews.checker_crew.checker_crew import CheckerCrew


# set logger
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class ResourceOptimizerState(BaseModel):
    sentence_count: int = 1
    poem: str = ""


class ResourceOptimizerFlow(Flow[ResourceOptimizerState]):

    @start()
    def extract_read(self):
        pass

    @router(extract_read)
    def analyze(self):
        # need a crew
        result = AnalystCrew()
        if result is True:
            return "detect"
        else:
            return "not detect"

    @listen(or_("detect", "failed"))
    def provide_solution():
        # need a crew
        result = SolutionCrew()
        pass

    @router(provide_solution)
    def check_result(self):
        # need a crew
        result = CheckerCrew()
        if result is True:
            return "success"
        else:
            return "failed"

    @listen(or_("success", "not detect"))
    def write_output_send_noti(self):
        pass

    # @listen(write_output_send_noti)
    # def implement(self):
    #     # need a crew
    #     pass


def kickoff():
    resource_optimizer_flow = ResourceOptimizerFlow()
    resource_optimizer_flow.kickoff()


def plot():
    resource_optimizer_flow = ResourceOptimizerFlow()
    resource_optimizer_flow.plot()


if __name__ == "__main__":
    kickoff()
