# itell-chatbot-sandbox

I'd recommend different environments for different chatbot strategies.

## Nemo Guardrails

These instructions are intended to be used on the LEARlab server.

1. `conda create -n ENV_NAME --file ENV.txt`
2. `conda activate ENV_NAME`
3. `nemoguardrails chat --config=config/hello_world` to start the chat with guardrails. Use `--verbose` to understand what is happening.

## vLLM

1. `conda create -n vllm --clone base`
2. `conda activate vllm`
3. `pip install vllm python-dotenv supabase markdown`