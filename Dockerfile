FROM python:3.11-slim

# Create a non-root user that HuggingFace Spaces requires
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user uvicorn

# Copy the application code
COPY --chown=user api.py .
COPY --chown=user data/clean/ ./data/clean/

# Expose port 7860 for HuggingFace Spaces
EXPOSE 7860

# Run uvicorn on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
