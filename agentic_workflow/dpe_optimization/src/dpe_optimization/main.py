
import os
from pydantic import BaseModel
from datetime import datetime, timedelta
import json

from crewai.flow import Flow, listen, start, router, or_, persist

from dpe_optimization.crews.analyst_crew.analyst_crew import AnalystCrew
from dpe_optimization.crews.solution_crew.solution_crew import SolutionCrew
from dpe_optimization.crews.checker_crew.checker_crew import CheckerCrew


# Set Logger
# logging.basicConfig()
# logging.getLogger().setLevel(logging.INFO)

# Global Variables
EXEC_DATE = datetime.today().strftime("%Y-%m-%d")
EXEC_UTC_DATE = datetime.utcnow().date().strftime("%Y-%m-%d")
FLOW_TIMER = datetime.now()


class ResourceOptimizerState(BaseModel):
    # Note: 'id' field is automatically added to all states
    # Structure State not supported dict type annotations: TypeError: Expected two type arguments for <class 'dict'>, got 1
    
    # General Metadata
    metadata: dict = {
        "execution_date": EXEC_DATE,
        "execution_date_utc": EXEC_UTC_DATE,
        "execution_time": None
    }
    
    # Crew Input
    flow_input: dict = {
        # "analyst": "Analyze financial data and conclude expense based on categories for data in year 2025"
        "analyst": "Aggregate Income Category based on given knowledge"
        # "analyst": "Get me the names of top 3 highest expense category from the database"
    }
    
    # Flow Output
    flow_output: dict = {
        "extract_read_task": {
            "execution_time": None,
            "output": None 
        },
        "analyze_task": {
            "execution_time": None,
            "output": None,
            "token_usage": None
        },
        # other flow tasks
    }

    def to_dict(self):
        return {
            "metadata": self.metadata,
            "flow_input": self.flow_input,
            "flow_output": self.flow_output,
        }

    def convert_execution_time(self):
        """Convert by applying -> lambda element: str(timedelta(seconds=<element>.seconds))"""
        self.metadata["execution_time"] = str(timedelta(seconds=self.metadata["execution_time"].seconds))
        self.flow_output["extract_read_task"]["execution_time"] = str(timedelta(seconds=self.flow_output["extract_read_task"]["execution_time"].seconds))
        self.flow_output["analyze_task"]["execution_time"] = str(timedelta(seconds=self.flow_output["analyze_task"]["execution_time"].seconds))
        return self


class ResourceOptimizerFlow(Flow[ResourceOptimizerState]):

    @start()
    def extract_read_task(self):
        """Extract Input to knowledge directories"""
        start_time = datetime.now()
        
        file_name = f"data_{EXEC_DATE}.csv"
        if os.path.isfile(path=f"knowledge/{file_name}"):
            print(f"{file_name} does exist, skipping extract")
        else:
            print(f"{file_name} does not exist, starting knowledge input extraction")
            # extract metrics to files by sql
            pass
        
        # log output
        end_time = datetime.now()
        self.state.flow_output["extract_read_task"]["execution_time"] = end_time - start_time

    @router(extract_read_task)
    def analyze(self):
        start_time = datetime.now()

        # need a crew
        inputs = {
            "exec_date": EXEC_DATE,
            "question": self.state.flow_input.get("analyst")
        }
        result = AnalystCrew(custom_inputs=inputs).crew().kickoff(inputs=inputs)
        end_time = datetime.now()
        
        # log output
        self.state.flow_output["analyze_task"]["execution_time"] = end_time - start_time
        self.state.flow_output["analyze_task"]["output"] = result.raw
        self.state.flow_output["analyze_task"]["token_usage"] = result.token_usage.__dict__
        
        # if result is True:
        #     return "detect"
        # else:
        #     return "not detect"

    # @listen(or_("detect", "failed"))
    # def provide_solution():
    #     # need a crew
    #     result = SolutionCrew()
    #     pass

    # @router(provide_solution)
    # def check_result(self):
    #     # need a crew
    #     result = CheckerCrew()
    #     if result is True:
    #         return "success"
    #     else:
    #         return "failed"

    # @listen(or_("success", "not detect"))
    @listen(analyze)
    def write_output_send_noti(self):
        self.state.metadata["execution_time"] = datetime.now() - FLOW_TIMER
        with open("state_output.json", "w") as f:
            json.dump(self.state.convert_execution_time().to_dict(), f, indent=4)
            f.close()
        print("Writing State File Output Success")

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
