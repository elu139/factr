# factr

https://factr-production.up.railway.app/

![There should be something here](images/factr icon.png)

factr is an AI-powered system designed to detect sophisticated misinformation by analyzing content across multiple modalities (text, audio, and video).

Current solutions are flawed. Human-powered systems like X's Community Notes are slow and inconsistent. Meanwhile, AI tools like McAfee's deepfake detector are too simplistic, often flagging content with any AI-generated component (e.g., AI audio over real video) as entirely fake.

Our solution offers a more nuanced approach. Instead of just detecting AI generation, factr cross-references visual, audio, and textual information within a piece of content to identify logical contradictions and inconsistencies—the true hallmarks of misinformation.

Ongoing Steps: Fine-tune the misinformation detection accuracy:
There’s no way to hit 0% error with an LLM-based detection system, but the most effective setup is a tight RAG pipeline with verification and abstention: retrieve from a high-quality corpus with hybrid search, make the model quote sources directly (and refuse if not found), use constrained decoding and tools for math/code, then add a verification loop or cross-model check. Combine that with confidence thresholds so the system can abstain instead of hallucinate. This drives error rates very low, but the only true “zero” comes from returning exact source text, or prompting it to return uncertain cases to a human.

Synthesize models of OpenCLIP, BLIP-2 LLaVA, and Instagram itself to improve factr LLM's ability to detect misinformation/anomalies/inconsistencies in Instagram feed posts, reels, carousels, stories, etc.

Focusing primarily on fine-tuning current detection algorithm -- current website deployment had to sacrifice CLIP model features due to space and dependency constraints. New chapter prioritizes model accuracy over ability to deploy (will start locally and move on towards deployment, most likely via AWS Lambda).

Turn factr from a website into a chrome extension
