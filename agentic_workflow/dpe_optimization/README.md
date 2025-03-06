# DPE Optimization Crew

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/dpe_optimization/crews/<crew-name>/config/agents.yaml` to define your agents
- Modify `src/dpe_optimization/crews/<crew-name>/config/tasks.yaml` to define your tasks
- Modify `src/dpe_optimization/crews/<crew-name>/<crew-name>.py` to add your own crew logic, knowledge, tools and specific args
- Modify `src/dpe_optimization/main.py` to add custom inputs/state/flow logic for your agents flow
- Add Database sqlite `./agentic_workflow/dpe_optimization/test.db` for reference information

## Running the Project

```bash
# work dir: ./agentic_workflow/dpe_optimization
crewai flow kickoff

# clear Knowledge cache
# export OPEN_API_KEY=""
crewai reset-memories --knowledge
# or remove manually by go to "C:/Users/<user>/AppData/Local/CrewAI/dpe_optimization/knowledge/"
# and delete chroma.sqlite3 file

# plot crewai workflow
crewai flow plot
# check 'crewai_flow.html'
```