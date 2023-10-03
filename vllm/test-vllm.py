from vllm import LLM
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

wording_clf = pipeline(
    "text-classification",
    model="tiedaar/longformer-wording-global",
    tokenizer=tokenizer,
    device = "cuda"
)

content_clf = pipeline(
    "text-classification",
    model="tiedaar/longformer-content-global",
    tokenizer=tokenizer,
    device = "cuda"
)

mpnet_classifier = pipeline(
    "text-classification", model="tiedaar/short-answer-classification", device="cuda"
)

bleurt_classifier = pipeline(
    'text-classification', model="vaiibhavgupta/finetuned-bleurt-large", device="cuda"
)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

llm = LLM(model="Open-Orca/OpenOrcaxOpenChat-Preview2-13B", gpu_memory_utilization=0.80)
print("*"*80)
print("models loaded")

for i in range(20):
    text = "sample"*i
    print(wording_clf(text))
    print(content_clf(text))
    print(mpnet_classifier(text))
    print(bleurt_classifier(text))
    print(embedding_model.encode(text)[0])
    print(llm.generate("Hello, my name is"))