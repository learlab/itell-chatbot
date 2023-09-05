# itell-chatbot-sandbox

I am experimenting with venv for this sandbox.

1. In the root directory of this project, open a terminal and run `conda deactivate`.
2. Run `source venv/bin/activate` to use the venv environment.
3. Run `export $(xargs < .env)` to get the OpenAI API key.
4. Run `nemoguardrails chat --config=config/hello_world` to start the chat with guardrails. Use `--verbose` to understand what is happening.