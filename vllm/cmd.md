curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Open-Orca/OpenOrcaxOpenChat-Preview2-13B",
        "prompt": "San Francisco is the capital of",
        "max_tokens": 128,
        "frequency_penalty": 0.05,
        "temperature": .8,
        "stream": "False",
        "n": 1,
        "use_beam_search": "True",
        "best_of": 4,
        "early_stopping": "True",
        "length_penalty": 0.5
    }'


curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Open-Orca/OpenOrcaxOpenChat-Preview2-13B",
        "prompt": "San Francisco is the capital of",
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": "False"
    }'