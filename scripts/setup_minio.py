"""Setup MinIO bucket."""
import os, logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_minio():
    s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "mlflow"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "mlflow123"))
    bucket = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
    try:
        s3.head_bucket(Bucket=bucket); logger.info(f"Bucket '{bucket}' exists")
    except ClientError:
        s3.create_bucket(Bucket=bucket); logger.info(f"Created '{bucket}'")


if __name__ == "__main__":
    setup_minio()
