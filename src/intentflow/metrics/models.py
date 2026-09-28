from pydantic import BaseModel


class MetricsResult(BaseModel):
    baseline: str
    num_benign: int = 0
    num_attacked: int = 0
    pcr: float = 0.0
    tmr: float = 0.0
    asr: float = 0.0
    btsr: float = 0.0
    pbr: float = 0.0
    fbr: float = 0.0
    latency_median: float = 0.0
    latency_p95: float = 0.0
