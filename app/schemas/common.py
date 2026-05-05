

from pydantic import BaseModel, Field, field_validator, model_validator


class HoldingInput(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    weight: float = Field(ge=0.0, le=1.0)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class PortfolioInput(BaseModel):
    holdings: list[HoldingInput] = Field(min_length=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "PortfolioInput":
        total = sum(h.weight for h in self.holdings)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Holding weights must sum to 1.0 (got {total:.4f})")
        return self
