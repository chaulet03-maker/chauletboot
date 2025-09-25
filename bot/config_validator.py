
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class LimitsModel(BaseModel):
    max_total_positions: int = Field(6, ge=1)
    max_per_symbol: int = Field(4, ge=1)
    no_hedge: bool = False

class FundingGuardModel(BaseModel):
    enabled: bool = False
    annualized_bps_limit: int = 5000

class TrailingModel(BaseModel):
    enabled: bool = True
    mode: str = Field("atr", pattern="^(atr|ema|percent)$")
    atr_k: float = 2.0
    ema_key: str = "ema_fast"
    ema_k: float = 1.0
    percent: float = 0.6
    min_step_atr: float = 0.5

class LearningModel(BaseModel):
    auto_throttle: bool = True
    memory_pauses: bool = True
    nudge_params: bool = False

class CorrelationGuardModel(BaseModel):
    enabled: bool = True
    clusters: List[List[str]] = [["BTC/USDT:USDT","ETH/USDT:USDT"]]
    same_side_max_exposure_ratio: float = 0.6

class FundingWindowModel(BaseModel):
    enabled: bool = True
    minutes: int = 7
    fr_abs_bps_min: float = 300.0

class ConfigModel(BaseModel):
    mode: str = Field("paper", pattern="^(paper|real)$")
    symbols: List[str] = ["BTC/USDT:USDT","ETH/USDT:USDT"]
    limits: LimitsModel = LimitsModel()
    funding_guard: FundingGuardModel = FundingGuardModel()
    trailing: TrailingModel = TrailingModel()
    learning: LearningModel = LearningModel()
    correlation_guard: CorrelationGuardModel = CorrelationGuardModel()
    funding_window: FundingWindowModel = FundingWindowModel()
    portfolio_caps: Dict[str, Any] = {}
    indicators: Dict[str, Any] = {}
    health: Dict[str, Any] = {}

def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model = ConfigModel(**cfg)
        return model.dict()
    except Exception as e:
        # Producir un mensaje claro en español
        raise ValueError(f"Config inválida: {e}")
