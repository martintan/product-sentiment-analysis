## product sentiment analysis

## setup

```bash
python3 -m venv env

# mac os
source ./env/bin/activate
# windows (cmd)
.\env\Scripts\activate.bat
# windows (powershell)
.\env\Scripts\Activate.ps1

pip install -r requirements.txt

# you may need to download nltk data, which requires certificates
# mac os (replace 3.x with your python version)
"/Applications/Python 3.x/Install Certificates.command"
# windows
pip install --upgrade certifi
```

## running the application

```bash
python main.py
```

## dataset

- `product_reviews_data.csv` - taken from Kaggle [Amazon Product Reviews](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset) dataset.