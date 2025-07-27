PROMPTS = {}

PROMPTS[
    'CoT_prompt'
] = """You are an intelligent assistant that helps a human analyst to analyze online news about its veracity. Your task is to 
give veracity prediction "True" OR "Fake" on news items to indicate they are true news OR fake news. 

Instructions: 
1. You are provided with a news including a title and several paragraphs of news content. 
2. Based on your knowledge, assess the factual accuracy of the news. 
3. Before presenting your conclusion, think through the process step-by-step. Include a summary of the key points from your knowledge 
as part of your reasoning. 
4. For the conclusion, output the final answer as a JSON object in the following format: 
{{
"prediction": "True" OR "False", 
"reasoning": "Your reasoning process."
}}

Given news content: 
news title: {news_title}
news article: {news_article}
publishing date: {date}
"""

