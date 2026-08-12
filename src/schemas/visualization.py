from pydantic import BaseModel
from typing import List

from src.schemas.sensor import SensorSample



class VisualizationRequest(BaseModel):
    samples: list[SensorSample]


class VisualizationPoint(BaseModel):
    x: float
    y: float
    z: float


class VisualizationMarker(BaseModel):
    index: int
    x: float
    y: float
    z: float


class VisualizationResponse(BaseModel):
    points: List[VisualizationPoint]
    markers: List[VisualizationMarker]