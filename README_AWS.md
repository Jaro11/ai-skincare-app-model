# AWS Deployment

This app runs Streamlit as the UI and serves the acne model from the same container.

## Recommended AWS Target

Use AWS App Runner first. It is simpler than ECS for a Streamlit app because it handles HTTPS, scaling, load balancing, and container hosting with less setup.

Use at least 2 GB memory. The app loads TensorFlow plus DeepFace, so very small instances can fail during startup or first image analysis.

## Local Container Test

```bash
docker build -t getglowmind .
docker run --rm -p 8501:8501 getglowmind
```

Open:

```text
http://localhost:8501
```

## Push To Amazon ECR

Replace the placeholders with your AWS values.

```bash
aws ecr create-repository --repository-name getglowmind
aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com
docker tag getglowmind:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/getglowmind:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/getglowmind:latest
```

## Create App Runner Service

1. Open AWS App Runner.
2. Create service.
3. Source: Container registry.
4. Provider: Amazon ECR.
5. Image URI: `YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/getglowmind:latest`.
6. Port: `8501`.
7. Memory: 2 GB minimum.
8. CPU: 1 vCPU minimum.
9. Deploy.

## Files Required In The Image

The Dockerfile copies only:

```text
app.py
predict.py
requirements.txt
packages.txt
acne_mobilenet_patch_model.h5
best_threshold.json
```

The training folders are not needed in production and are excluded by `.dockerignore`.

## Notes

DeepFace may download model weights the first time face analysis runs in a fresh container. That can make the first uploaded image slower. The app delays DeepFace loading until an image is uploaded so AWS health checks can pass quickly.
