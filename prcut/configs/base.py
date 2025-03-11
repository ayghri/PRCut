from pathlib import Path
import pydantic
import json


class BaseModel(pydantic.BaseModel):

    def __str__(self):
        return json.dumps(self.model_dump(), indent=2)

    def __repr__(self):
        return str(self)

    def save(self, path: str | Path):
        pass
