from fastapi.testclient import TestClient
from api_v2.main import app, target_month

client = TestClient(app)
BASE = {'market': 'Example market', 'current_price': 60, 'origin_year': 2026, 'origin_month': 11}

def test_health():
    r=client.get('/health'); assert r.status_code==200 and r.json()['ready']

def test_models():
    r=client.get('/models'); assert r.status_code==200
    assert set(r.json()['horizons']) == {'1','2','3'}

def test_forecast_all_horizons():
    for h in (1,2,3):
        r=client.post('/predict',json={**BASE,'horizon':h})
        assert r.status_code==200, r.text
        p=r.json(); assert p['predicted_price']==60 and p['model']=='persistence'
        assert p['target_date']==('2026-12' if h==1 else f'2027-{h-1:02d}')
        assert p['interval_80']['lower'] <= 60 <= p['interval_80']['upper']

def test_reject_invalid():
    for payload in ({**BASE,'horizon':4},{**BASE,'horizon':1,'current_price':-5},{**BASE,'horizon':1,'unrecognized':'value'}):
        assert client.post('/predict',json=payload).status_code==422

def test_target_month_rollover():
    assert target_month(2026,12,3)==(2027,3)
