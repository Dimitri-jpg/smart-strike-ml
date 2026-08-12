from pydantic import BaseModel
from typing import List



class SensorData(BaseModel):
    samples: list


class SensorSample(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float

    linear_x: float
    linear_y: float
    linear_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    rot_x: float
    rot_y: float
    rot_z: float
    rot_w: float