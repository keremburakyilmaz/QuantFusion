from fastapi import FastAPI
from routers import trading, risk, portfolio, derivatives, sentiment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QuantFusion Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trading.router, prefix="/trading")
app.include_router(risk.router, prefix="/risk")
app.include_router(portfolio.router, prefix="/portfolio")
app.include_router(derivatives.router, prefix="/derivatives")
app.include_router(sentiment.router, prefix="/sentiment")
