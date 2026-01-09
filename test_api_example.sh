#!/bin/bash
# Example usage of the test_inference_api.py script

# Make sure requests is installed
pip install requests

# Test with a small sample (10 examples) using NatSQL
python3 test_inference_api.py \
    --api-url "http://98.80.127.45:8000/infer" \
    --data-file "NatSQL/NatSQLv1_6/train_spider.json" \
    --target-type "natsql" \
    --limit 10 \
    --verbose

# Test with SQL target type
# python3 test_inference_api.py \
#     --api-url "http://98.80.127.45:8000/infer" \
#     --data-file "data/spider/train_spider.json" \
#     --target-type "sql" \
#     --limit 10 \
#     --verbose

# Test and save results to file
# python3 test_inference_api.py \
#     --api-url "http://98.80.127.45:8000/infer" \
#     --data-file "NatSQL/NatSQLv1_6/train_spider.json" \
#     --target-type "natsql" \
#     --limit 100 \
#     --output "test_results.json"
