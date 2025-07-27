import json


def remove_labels(article):
    pf_label_keywords = ['true', 'false', 'pants']
    tmp_split = article.split('.')
    tmp_article_without_label = ''
    for i in range(len(tmp_split)):
        for keyword in pf_label_keywords:
            if keyword in tmp_split[-i - 1].lower():
                tmp_article_without_label = '.'.join(tmp_split[:-i - 1])
                tmp_article_without_label += '.'
                return tmp_article_without_label
            
def load_liar_new():
    liar_new_path = './data/LIAR-New/LIAR-New.jsonl'
    liar_new_article_path = './data/LIAR-New/LIAR-New_articles.jsonl'
    liar_new, liar_new_article = {}, {}

    with open(liar_new_article_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            index, article = line['example_id'], line['article_text']
            # liar_new_article[index] = article
            liar_new_article[index] = remove_labels(article)

    with open(liar_new_path, 'r', encoding='utf-8') as f:
        # data = [json.loads(line) for line in f]
        for line in f:
            line = json.loads(line)
            index = line['example_id']
            date = line['date']
            label = line['label']
            title = line['statement']
            article = liar_new_article[index]

            # assemble
            liar_new[index] = {
                'title': title,
                'date': date, 
                'article': article,
                'label': label
            }
    return liar_new

