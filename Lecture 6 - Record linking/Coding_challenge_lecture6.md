# Week 6: Record Linking and Entity Matching

In this week's coding challenge, you are tasked with solving a record-linking problem using the provided datasets. This problem requires you to identify relationships between the financial data and the news articles. We will start with a simple exercise to warm up and then progress to the main linking challenge.

The data to be used is simulated data, which can be found [here](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/main/Lecture%206%20-%20Record%20linking/Data/Large_linking_toydata)

---

### Part 1: Finding Records for "NVDA"

**Task:**
1. Load the financial and news data from the provided CSV files.
2. Filter the financial data for all records where the ticker is "NVDA"
3. In the news data, find all articles mentioning "NVIDIA" as the company.

**Questions:**
- How many records for "NVDA" are present in the financial dataset?
- How many articles mention "NVIDIA" in the news dataset?

---

### Part 2: Full Record Linking Challenge

**Goal:**
Build a system to link financial records to related news articles. Please consider data quality. 

**Questions:**
- How many financial records were successfully linked to news articles?
- Are there financial records that do not have matching news articles? Why might this occur?
- Are there news articles that do not link to any financial record? Why might this occur?

### Hint:
Have you checked whether the company name column is correct?
