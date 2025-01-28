# Agentic Workflow

*Patcharanat P.*

## Installation

```bash
# Activate Python Virtualenv
python venv pyenv
# source pyenv/Scripts/activate

# Install Dependencies
make install

# crewai create crew <project_name>
crewai create crew poc
```

```bash
.
|-- README.md
|   
+---poc
    |-- .env
    |-- pyproject.toml
    |-- README.md
    |   
    +---knowledge
    |   +-- user_preference.txt
    |       
    +---src
    |   +---poc
    |       |-- crew.py
    |       |-- main.py
    |       |
    |       +---config
    |       |   |-- agents.yaml
    |       |   +-- tasks.yaml
    |       |       
    |       +---tools
    |           +-- custom_tool.py
    |               
    +---tests
```