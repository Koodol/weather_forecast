from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pandas as pd
import io
from datetime import datetime, timedelta
import joblib
import numpy as np

app = FastAPI()

# 모델 로드 (실제 경로에 있는 모델 파일로 변경하세요)
model = joblib.load('weather_forecast_model.pkl')

# 기상청 API 정보
KMA_API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"

class PredictionRequest(BaseModel):
    stn: int  # 기상청 지역 코드 (stn 코드)
    date: str  # 예측을 원하는 날짜 (YYYY-MM-DD 형식)

def fetch_weather_data(stn, date, num_days=10):
    end_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=num_days)

    params = {
        "tm1": start_date.strftime("%Y%m%d"),
        "tm2": end_date.strftime("%Y%m%d"),
        "stn": stn,
        "authKey": "8hLO_pajRsSSzv6Wo_bEIg"  # 기상청 API 키를 입력하세요
    }

    response = requests.get(KMA_API_URL, params=params)
    if response.status_code == 200:
        column_names = [
            "TM", "STN", "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM", "WD_INS",
            "WS_INS", "WS_INS_TM", "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM",
            "TD_AVG", "TS_AVG", "TG_MIN", "HM_AVG", "HM_MIN", "HM_MIN_TM", "PV_AVG", "EV_S",
            "EV_L", "FG_DUR", "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM", "PS_MIN", "PS_MIN_TM",
            "CA_TOT", "SS_DAY", "SS_DUR", "SS_CMB", "SI_DAY", "SI_60M_MAX", "SI_60M_MAX_TM",
            "RN_DAY", "RN_D99", "RN_DUR", "RN_60M_MAX", "RN_60M_MAX_TM", "RN_10M_MAX",
            "RN_10M_MAX_TM", "RN_POW_MAX", "RN_POW_MAX_TM", "SD_NEW", "SD_NEW_TM", "SD_MAX",
            "SD_MAX_TM", "TE_05", "TE_10", "TE_15", "TE_30", "TE_50"
        ]
        
        # 데이터를 DataFrame으로 변환
        data = pd.read_csv(
            io.StringIO(response.text),
            sep='\s+',
            comment='#',
            names=column_names,
            header=None,
            skiprows=1
        )
        
        return data
    else:
        raise HTTPException(status_code=400, detail="Failed to fetch weather data from KMA API")

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)].values
        xs.append(x)
    return np.array(xs)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # 과거 10일간의 기상 데이터 가져오기
        weather_data = fetch_weather_data(request.stn, request.date)
        
        # 시퀀스 길이 정의
        seq_length = 10
        
        # 시퀀스 생성
        sequences = create_sequences(weather_data[['TA_AVG']], seq_length)
        
        # 예측 수행
        prediction = model.predict(sequences)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
