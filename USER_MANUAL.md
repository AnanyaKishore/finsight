# FinSight — User manual

FinSight shows ten large US stocks on one web page:
- where each price has been;
- where a model thinks it may go over the next two weeks;
- whether recent news about the company is positive or negative;
- one suggested way to split money across the ten stocks.

This manual is for anyone who uses FinSight. You don't need any technical knowledge. The same
help is inside the app: click **Help** at the top of any page.

> **Important.** FinSight is a learning project. Its forecasts and scores are estimates and
> are often wrong. Nothing in it is a recommendation to buy or sell anything.

---

## 1. Opening FinSight

FinSight runs on the computer it was installed on. Whoever set it up only needs to start it
once after the computer turns on (section 7). Then:

1. Open a web browser (Chrome, Firefox, Edge or Safari).
2. Go to **http://localhost:8080**

The overview page opens. If it says the server cannot be reached, see section 6.

## 2. The overview page

![Overview page](docs/images/manual/01-overview.png)

What you see, from top to bottom:

| Part | What it is |
|---|---|
| **Search box** (top) | Find a stock by name, ticker or product (section 3). |
| **Status dot** (top right) | Green: everything works. Amber: one part is down. Red: the server cannot be reached. Click it for details. |
| **Yellow banner** | A reminder that this is not investment advice. |
| **The ten stocks** | Apple, Microsoft, Alphabet, Amazon, Tesla, NVIDIA, Meta, JPMorgan Chase, Walmart and Visa, each with its latest price and daily change. Click one to open its page. |
| **Optimised portfolio** | A suggested split of money across the ten stocks (section 5). |
| **Try the sentiment model** | Paste headlines and see how the model scores them (section 4.4). |

Green with an upward arrow means the price rose; red with a downward arrow means it fell.
Prices are end-of-day closing prices, not live prices.

## 3. Finding a stock

![Search](docs/images/manual/02-search.png)

1. Click the search box, or press the **/** key.
2. Start typing a company name ("apple"), a ticker ("AAPL") or a product ("iPhone").
3. Click a suggestion, or use the **↓** and **↑** keys and press **Enter**. **Esc** closes the list.

If you type a stock FinSight does not cover, it says so.

## 4. The stock page

![Stock page](docs/images/manual/03-stock.png)

### 4.1 Overview
The company name, ticker and sector, the latest closing price, and the change from the day
before.

### 4.2 Price history
- Choose a period with the buttons: **1W** (one week), **1M**, **3M**, **6M** or **1Y** (one year).
- Hover over the line to read the price on any day.
- The sentence under the chart gives the change over the chosen period.
- **Show the prices as a table** lists every value, which is useful with a screen reader or for copying.

### 4.3 14-day forecast

![Forecast](docs/images/manual/04-forecast.png)

- The **solid line** is the real closing price.
- The **dashed line** is the model's estimate for the next 14 trading days.
- The text under the chart says what the model expects, and **how accurate it was in testing**
  compared with simply assuming the price will not change. If the two numbers are close,
  treat the forecast as a rough guide only.
- If the forecasting part of the system is down, you will see the **last saved forecast**,
  clearly marked as a saved copy.

### 4.4 News sentiment

![News sentiment](docs/images/manual/05-sentiment.png)

- Each **bar is one week** of news about the company.
  - Bars **above** the middle line mean the news was mostly positive.
  - Bars **below** it mean mostly negative.
  - Taller bars mean stronger.
- The small chart underneath shows **how many articles** there were each week. A week with
  few articles is less reliable.
- **Click a bar** to see the most positive and most negative headlines of that week, with links
  to the articles. This shows *why* the week scored as it did.
- Choose **4w**, **12w** or **26w** to see more or fewer weeks.

**Trying the sentiment model yourself** (on the overview page):

![Try the sentiment model](docs/images/manual/06-analyse.png)

1. Paste one or more headlines, one per line.
2. Press **Analyse** (or **Ctrl+Enter**).
3. Each line gets a label (positive, negative or neutral), a score from −1 (very negative)
   to +1 (very positive), and the model's confidence.

## 5. The optimised portfolio

![Portfolio](docs/images/manual/07-portfolio.png)

The pie chart shows how a sum of money could be split across the ten stocks to get the best
balance of return and risk over the **last 30 trading days**. It is recalculated once a day.

| Number | Meaning |
|---|---|
| **Sharpe ratio** | Return earned per unit of risk. Higher is better. |
| **Expected return** | The yearly return the last 30 days suggest. |
| **Volatility** | How much the portfolio's value swings, per year. Lower is calmer. |

**Read the reality check under the chart before trusting the pie.** It repeats the same
method week by week over the past year and compares it with simply splitting money equally.
Often the equal split did better, which is a well-known result.

## 6. When something is not working

| What you see | What it means | What to do |
|---|---|---|
| Amber status dot | One part (for example the forecasting model) is down; the rest works | Click the dot to see which part. Forecasts fall back to the last saved copy. Usually it recovers by itself within a minute. |
| Red dot, "cannot reach the server" | The FinSight services are not running | Ask whoever installed FinSight to start it (section 7). |
| "Could not load" on one section | That section failed; others still work | Press **Try again**. |
| "Not covered" page | The stock is not one of the ten | Go back to the overview and pick one of the ten. |
| Forecast says "saved copy" | The forecasting model is restarting or down | The saved forecast is from an earlier run; it updates automatically when the model is back. |
| Prices look a day old | Prices are end-of-day and are updated each weekday morning (06:30 India time) | Nothing; this is normal. On a weekend or holiday the last trading day is shown. |

![Status panel](docs/images/manual/08-status.png)

## 7. Starting and stopping FinSight (for whoever installed it)

Open a terminal in the FinSight folder:

```bash
docker compose up -d        # start everything (about one minute)
docker compose ps           # every service should say "running" or "healthy"
docker compose stop         # stop everything; your data is kept
```

FinSight keeps itself up to date: every weekday morning it fetches new prices and news and
refreshes the forecasts. If the computer was asleep at that time, it does this as soon as it
wakes.

## 8. The Pipeline page (for operators)

![Pipeline Console](docs/images/manual/09-pipeline.png)

The **Pipeline** link at the top opens the console that shows how FinSight keeps itself up to date.

| Card | What it shows |
|---|---|
| **Scheduler** | Whether a run is in progress, which step it is on, and when the next run is due. **Run now** starts the daily update; **Retrain now** retrains the forecasting model and lets the automatic check decide whether the new model replaces the current one. Both ask you to confirm. |
| **Pipeline stages** | Each box is one step of the data and model pipeline; arrows show which feeds which. Green succeeded, red failed, blue running, grey not run yet; an amber dot means its inputs changed since it last ran. Click a box for details. |
| **Run history** | One row per run. Click a row to see each step, any error message and the last lines of its log. "Partial" means the run finished but a non-essential step had a problem. |
| **Run duration, Throughput** | Whether runs are getting slower over time. Every chart has a table view. |
| **Monitoring** | Alerts firing now, and whether the recent data still looks like the data the model learned from. If many stocks change behaviour, or the model becomes clearly worse than a simple "price stays the same" guess, FinSight retrains itself (at most once a day). The **Grafana** link opens the same numbers as charts over time. |

![Monitoring](docs/images/manual/10-monitoring.png)

## 9. Accessibility and devices

- Works on phones and tablets: the layout adapts to the screen.
- Follows your system's light or dark mode.
- Fully usable with the keyboard: **/** to search, **Tab** to move, **Enter** to open.
- Every chart has a **table view** with the same numbers, for screen readers.

## 10. Glossary

| Term | Meaning |
|---|---|
| Closing price | The last price of the trading day. |
| Ticker | The short code of a stock, for example AAPL for Apple. |
| Forecast | The model's estimate of future closing prices. |
| MAPE | Average forecast error, as a percentage of the price. Lower is better. |
| Sentiment | Whether news is positive or negative, scored from −1 to +1. |
| Sharpe ratio | Return per unit of risk. |
| Volatility | How much a price swings. |
| Drift | Recent data no longer looking like the data a model learned from. |
