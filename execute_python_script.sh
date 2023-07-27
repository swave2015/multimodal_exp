#!/bin/bash

# Set the maximum number of retries
max_retries=100
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    # Execute the Python command
    python3 test_vlucan.py

    # Check the exit code of the Python command
    if [ $? -eq 0 ]; then
        echo "Python script executed successfully."
        break
    else
        echo "Python script failed. Retrying..."
        retry_count=$((retry_count + 1))
        sleep 5  # Add a small delay before retrying (optional)
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo "Python script failed after $max_retries retries. Exiting."
fi
