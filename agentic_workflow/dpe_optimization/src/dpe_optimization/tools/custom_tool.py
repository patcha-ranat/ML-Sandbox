from typing import Type
import sqlite3

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# For Example
# from crewai_tools import (
#     DirectoryReadTool,
#     FileReadTool,
#     SerperDevTool,
#     WebsiteSearchTool
# )


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."


class DatabaseQueryToolInput(BaseModel):
    database: str
    query: str


class DatabaseQueryTool(BaseTool):
    name: str = "DatabaseQueryTool"
    description: str = "Retrieve sql query string as input and execute on database and return output"
    args_schema: Type[BaseModel] = DatabaseQueryToolInput
    database: str = None
    query: str = None

    def __init__(self, database: str = None, query: str = None):
        super().__init__()
        self.database = database
        self.query = query

    def transform_query(self) -> str:
        """Read sql file spcified path to be a query or return the query if given input is already an sql query string"""
        if ".sql" in self.query:
            with open(self.query, "r") as f:
                sql_query = f.read()
                f.close()
        else:
            sql_query = self.query

        return sql_query
    
    def execute_query(self, sql_query: str):
        """Execute SQL query with target database and return the result"""
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        try:
            cursor.executescript(sql_query) # can execute multiple SQL statements in one call
            results = cursor.fetchall()
        except Exception as e:
            raise e
        finally:
            conn.close()

        return results

    def _run(self):
        """Entrypoint"""
        sql_query = self.transform_query()
        result = self.execute_query(sql_query=sql_query)
        return result
