# Instagram API Configuration
INSTAGRAM_ACCESS_TOKEN=your_instagram_access_token_here
INSTAGRAM_CLIENT_ID=your_instagram_client_id_here
INSTAGRAM_CLIENT_SECRET=your_instagram_client_secret_here

# AWS Configuration (for deployment and additional services)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=factr-ai-data

# Application Configuration
DEBUG=False
LOG_LEVEL=INFO
MAX_CONTENT_SIZE=10485760  # 10MB limit for uploaded content

# ML Model Configuration
MODEL_CACHE_DIR=./models
CLIP_MODEL_NAME=ViT-B/32
BERT_MODEL_NAME=bert-base-uncased

# Rate Limiting
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=3600  # 1 hour

# Database (we might add this later for storing analysis results)
# DATABASE_URL=postgresql://user:password@localhost:5432/factr_ai

# External Services
REVERSE_IMAGE_SEARCH_API_KEY=your_reverse_search_api_key
FACT_CHECK_API_KEY=your_fact_check_api_key