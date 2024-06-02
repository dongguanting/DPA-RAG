## Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation</h2>

<p>
📃 <a href="">ArXiv Paper</a>

</p>


⭐ **We will release the templates, and codes within a month. Thanks for your attention!**

## Introduction
 we propose a \underline{D}ual \underline{P}reference \underline{A}lignment for \underline{R}etrieval-\underline{A}ugmented \underline{G}eneration (DPA-RAG), a universal framework designed to align diverse preference knowledge within RAG systems. DPA-RAG consists of three key components: 
(1)~\textbf{Preference Knowledge Construction}: motivated by our preliminary results, we first extract the specific knowledge that significantly affects LLMs' reasoning preferences. Then we introduce five query augmentation strategies and a quality filtering process to synthesize high-quality preference knowledge. 
(2)~\textbf{Reranker-LLM Alignment}: To meet the diverse LLMs' knowledge preferences, we carefully design multi-grained alignment tasks for fine-tuning a preference-aligned reranker. Specifically, we jointly integrate pair-wise, point-wise, and contrastive preference alignment abilities into the reranker via multi-task optimization~\citep{sener2018multi}. By this means, the reranker could provide the necessary knowledge for LLM's inference, achieving external alignment between retriever and LLMs.
(3)~\textbf{LLM Self-Alignment}: To further enable LLMs to concentrate on knowledge aligned with their reasoning preferences, we introduce a pre-aligned phrase prior to the vanilla SFT stage. This stage allows LLMs to capture preference-aligned knowledge from multiple documents, completing the LLM's internal self-alignment.

Our code will be released soon!
