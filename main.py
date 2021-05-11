import pandas as pd
from sentiment_analyzer import build_net, train, prepare_data, get_vocab_len, PRED_2_LABEL
from yahoo_crawler import get_delta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("TKAgg")


def load_stocks(companies, tickers):
    stocks = defaultdict(lambda: defaultdict())
    companies_len = len(companies)
    for i in range(companies_len):
        company = companies[i]
        stocks[company]['ticker'] = tickers[i]
        stocks[company]['enumerated_indices'] = []
        stocks[company]['ordered_sentiments'] = []
        stocks[company]['percent_change'] = get_delta(stocks[company]['ticker'])

    return stocks


all_data = pd.read_csv('full_dataset.csv')
all_data = all_data[all_data.y != 0] # remove neutral posts
all_data['y'] -= 1 # update for categorical

vocab_len = get_vocab_len(all_data)
x_train, y_train, x_dev, y_dev, tokenizer, x_dev_text = prepare_data(vocab_len, all_data['title'], all_data['y'])


# optimizers = ['Adam_W', 'RMSprop', 'SGD', 'Adadelta', 'Adagrad']
optimizers = ['Adam_W', 'Adam']
# l_styles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10))]
l_styles = ['solid', 'dotted']
# optimizers = ['Adam_W']
# print(x_dev_text)


epochs = 30

fig_f1, ax_f1 = plt.subplots()
ax_f1.set_title('F1 score')
ax_f1.set_ylabel('F1')
ax_f1.set_xlabel('epoch')
ax_f1.set_xticks(np.arange(0, epochs))

fig_loss, ax_loss = plt.subplots()
ax_loss.set_title('Model Loss')
ax_loss.set_ylabel('loss')
ax_loss.set_xlabel('epoch')
ax_loss.set_xticks(np.arange(0, epochs))

f1_lines = []
loss_lines = []
for optim_idx, optimizer in enumerate(optimizers):
    print("-----------------------------------")
    print(f"Current optimizer: {optimizer}")
    model = build_net(tokenizer.word_index, optimizer)
    train(model, epochs, optimizer, l_styles[optim_idx], x_train, y_train, x_dev, y_dev, ax_metric=ax_f1, ax_loss=ax_loss)
    print("-----------------------------------")


ax_f1.legend(loc='upper right', frameon=True)
ax_loss.legend(loc='upper right', frameon=True)

plt.show()

# companies['APPLE']['chronological_indices'] = []
# companies['APPLE']['sentiment'] = []
companies_ = ['APPLE', 'TESLA', 'CARNIVAL', 'AMC', 'FACEBOOK', 'PALANTIR', 'CORSAIR', 'EXXON', 'Sundial']
tickers_ = ['AAPL', 'TSLA', 'CCL', 'AMC', 'FB', 'PLTR', 'CRSR', 'XOM', 'SNDL']
stocks_ = load_stocks(companies_, tickers_)

for idx, title in enumerate(x_dev_text):

    title = title.split(" ")
    for org_name, values in stocks_.items():
        ticker = values['ticker']
        for word in title:
            word = word.upper()
            if org_name in word or ticker in word:
                stocks_[org_name]['enumerated_indices'].append(idx)
                sentiment_pred = model.predict_classes(x_dev[idx])[-1]
                stocks_[org_name]['ordered_sentiments'].append(PRED_2_LABEL[sentiment_pred])

                if ticker == 'TSLA' and idx in [7779, 7163, 6240, 2366]:
                    print(title, idx, PRED_2_LABEL[sentiment_pred])


                # if ticker == 'TSLA' and 'worse?' in title:
                #     print(title, 'TSLA', idx, 'ground truth:', PRED_2_LABEL[y_dev[207]])
                #     print(sentiment_pred, PRED_2_LABEL[sentiment_pred])

for org_name, values in stocks_.items():
    count = Counter(stocks_[org_name]['ordered_sentiments'])
    org = stocks_[org_name]
    print(f"--------{org_name}, {org['ticker']}--------")
    print(count.most_common())
    stocks_[org_name]['sentiment'] = count.most_common(1)[0][0]
    stocks_[org_name]['sentiment_count'] = count.most_common(1)[0][1]



    print(f"Ground Truth: {org['percent_change']}%")
    print(f"Prediction: sentiment = {org['sentiment']}, freq = {org['sentiment_count']}")

# print(companies)
#
# AAPL/Apple, TSLA/Tesla, CRSR/Corsair, AMC/AMC, CCL/Carnival FB/Facebook,  PLTR/Palantir


