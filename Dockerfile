FROM python:3.11.4


WORKDIR /src

# requirements.txtをコピー
COPY requirements.txt .

# ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコードをコピー
COPY . .

# uvicornのサーバーを立ち上げる
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000",  "--reload"]