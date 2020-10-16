FROM python:3.6

WORKDIR /tutorial

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["jupyter",  "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

