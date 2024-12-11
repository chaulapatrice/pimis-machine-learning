from neuralprophet import NeuralProphet
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import os

access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')


def generate_presigned_url(bucket_name: str, object_key: str, expiration=1200):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key,
            },
            ExpiresIn=expiration  # Expiration time in seconds
        )

        return presigned_url
    except (NoCredentialsError, PartialCredentialsError) as e:
        print("Error: AWS credentials not configured properly.", e)


def upload_file(file_name: str, s3_prefix: str):
    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    bucket_name = 'pimis-ml'
    key_name = f'{s3_prefix}/{file_name}'

    s3.upload_file(file_name, bucket_name, key_name)

    print("File uploaded successfully!")


def predict():
    url = generate_presigned_url("pimis-ml", "reports/export.xlsx")
    df = pd.read_excel(url)
    df['Date Created'] = pd.to_datetime(df['Date Created'])
    df = df.sort_values('Date Created')

    df = df[['Date Created', 'Total Revenue']]
    df.columns = ['ds', 'y']

    grpd = df.groupby('ds').sum()

    df = grpd.reset_index()
    df = df.sort_values('ds')

    model = NeuralProphet(trend_reg=0.5, seasonality_reg=0.5)
    model.fit(df, freq='D', early_stopping=True)

    future = model.make_future_dataframe(df, periods=365)
    forecast = model.predict(future)

    actual_prediction = model.predict(df)

    actual_prediction.set_index('ds', inplace=True)
    actual_prediction_monthly = actual_prediction.resample('ME').sum()
    actual_prediction_monthly.to_excel("results.xlsx", sheet_name='Actual Prediction')

    df.set_index('ds', inplace=True)
    actual_monthly = df.resample('ME').sum()
    actual_monthly.to_excel("results.xlsx", sheet_name='Actual')

    forecast.set_index('ds', inplace=True)
    forecast_monthly = forecast.resample('ME').sum()
    forecast_monthly.to_excel("results.xlsx", sheet_name="Forecast")

    plt.figure(figsize=(20, 5))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.plot(actual_prediction_monthly.index, actual_prediction_monthly['yhat1'], label='Actual Prediction', c='r',
             marker='o')
    plt.plot(forecast_monthly.index, forecast_monthly['yhat1'], label='Future Prediction', c='b', marker='o')
    plt.plot(actual_monthly.index, actual_monthly['y'], label='Actual', c='g', marker='o')
    plt.legend()

    plt.savefig("graph.pdf")

    upload_file("graph.pdf", "results")
    upload_file("results.xlsx", "results")


if __name__ == '__main__':
    predict()
