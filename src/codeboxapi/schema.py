"""
This file contains the schema for the CodeBox API.
It is used to validate the data returned from the API.
It also helps with type hinting and provides a nice
interface for interacting with the API.
"""

from typing import Optional
import pandas as pd
from io import StringIO

from pydantic import BaseModel


class CodeBoxStatus(BaseModel):
    """
    Represents the status of a CodeBox instance.
    """

    status: str

    def __str__(self):
        return self.status

    def __repr__(self):
        return f"Status({self.status})"

    def __eq__(self, other):
        return self.__str__() == other.__str__()


class CodeBoxOutput(BaseModel):
    """
    Represents the code execution output of a CodeBox instance.
    """

    type: str
    content: str

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"{self.type}({self.content})"

    def __eq__(self, other):
        return self.__str__() == other.__str__()


class CodeBoxFile(BaseModel):
    """
    Represents a file returned from a CodeBox instance.
    """

    name: str
    content: Optional[bytes] = None

    @classmethod
    def from_path(cls, path: str) -> "CodeBoxFile":
        if not path.startswith("/"):
            path = f"./{path}"
        with open(path, "rb") as f:
            path = path.split("/")[-1]
            return cls(name=path, content=f.read())
        
    def to_dataframe(self) -> pd.DataFrame:
        data_string = self.content.decode('utf-8')
        data_io = StringIO(data_string)
        df = pd.read_csv(data_io)
        return df

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"File({self.name})"
