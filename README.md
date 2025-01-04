# LLM_rumor
Rumor detection with LLM prompt engineering

## 1. Chain of propagation 

In Graph-based misinformation detection [1](https://arxiv.org/abs/2307.12639)., the propagation is frequently utilized for short social media news posts. In the propagation, the source news post has some comments under it, and the comments can serve as additional evidence when the source news post does not contain enough context to make a sure prediction. 

In a recent paper [2](https://arxiv.org/abs/2402.03916), the propagation is described in natural language and forwarded to the ChatGPT-3.5 in a prompt way, as follows. 

For example, we have a source 'news' and some 'comments'. They are packed in one prompt: 

'There is a piece of news: '+ news + 'There are comments for the news: ' + comments + 'You need to do: \
    (1) Based on the writing style and the commonsense knowledge, estimate the credibility of the news. \
    (2) Based on the comments, analyze whether there are any rebuttals or conflicts, and then accordingly verify the authenticity of the news. \
        Based on above results, please choose the answer from the following options: A. Fake, B. Real.' 

This prompt is delivered to the ChatGPT for a prediction on the veracity of the news as fake or real. When the comments are huge, GPT cannot handle those many tokens, then the comments are broken into chunks, and each chunk is input into the GPT. Besides, the GPT output from the last chunk is also delivered in the current chunk. More details are shown in the paper and the code implementation. 

To run the chain of propagation detection, simply: 

`
python chain_of_prop.py
`

The results are stored locally, which can be used for evaluation. 


## Reference

1. Gong S, Sinnott R O, Qi J, et al. Fake news detection through graph-based neural networks: A survey[J]. arXiv preprint arXiv:2307.12639, 2023. [link](https://arxiv.org/abs/2307.12639)
2. Liu Q, Tao X, Wu J, et al. Can Large Language Models Detect Rumors on Social Media?[J]. arXiv preprint arXiv:2402.03916, 2024. [link](https://arxiv.org/abs/2402.03916)
