# Attitudes of the Chinese Government Toward Internal Migrants: An Analysis of Evidence from The People's Daily

## Project Overview
This research project delves into the evolving dynamics of internal migration in China, particularly within the constraints of the hukou system—a household registration policy that limits the mobility and rights of Chinese citizens. Despite such restrictions, internal migration has seen a significant uptick due to economic incentives and enforced relocations, like those for large-scale infrastructure developments.

The study focuses on analyzing content from The People's Daily, the official newspaper of the Communist Party of China (CPC), from 1980 to 2023. The primary goal is to decode the topics and sentiments expressed about internal migration, examining how they reflect broader governmental attitudes and policy shifts over the decades.

## Research Objectives
- **Sentiment Analysis:** Utilize a custom-designed model that integrates RoBERTa and BiGRU to effectively process extended texts in Chinese, aiming to classify the underlying sentiments in the articles.
- **Topic Modeling:** Apply LDA (Latent Dirichlet Allocation) to identify prevalent themes and track their evolution over time within the dataset.

## Methodology
- **Sentiment Classification:**
The sentiment classification part of the project employs a hybrid model combining the robustness of RoBERTa with the sequence processing capabilities of BiGRU. This model is specifically tuned to handle the complexities of lengthy Chinese texts.

- **Topic Modeling:**
For topic analysis, the LDA model is used to extract and monitor thematic trends, providing insights into the shifts in discourse surrounding internal migrants and related policies.

## Data
- **People's Daily:**  The data from 1946-2003 were extracted from the People’s Daily archive available at https://www.laoziliao.net/. Data spanning 2018 to 2023 were collected from the official People’s Daily website at http://www.people.com.cn/. For the years 2004-2017, the dataset was built through download from Peking University People’s Database.
- **Model Training:** The training data is a collection of Chinese online news with sentiment label, obtained from https://www.datafountain.cn/competitions/350/datasets. 

## Data Descpritive
- **Presence of Total News Over Years:**
The graph below depicts the count of total news reports from 1970 to 2023.
<p align="center">
  <img src="/Plot/distribution_of_total_news_over_year.png" alt="Presence of Total News Over Year" width="800"/>
</p>


- **Presence of Internal Migration Related News Over Years:** 
The graph below illustrates the frequency of news reports related to internal migration in China from 1970 to 2023. The dataset was curated using a comprehensive lexicon of keywords relevant to internal migration, including "移民" (immigrants), "流动人口" (floating population), "外地人" (outsiders), "民工" (migrant workers), "进城务工人员" (urban migrant workers), "新市民" (new citizens), "盲流" (blind migration), "随迁子女" (migrating children), "落户" (settlement), "户口迁移" (household registration transfer), "人口迁移" (population migration), "流浪乞讨人员" (vagrant and beggar population), and "城乡移民" (urban-rural migration). To enhance the reliability of the analysis, only entries where keywords appeared more than twice were retained. Additionally, any texts containing names of foreign countries were excluded to maintain focus on domestic migration dynamics.
<p align="center">
  <img src="Plot/distribution_of_filtered_total_news_over_date.png" alt="Presence of Related News Over Year" width="600"/>
</p>p

## Results and Discussion
The findings of this study highlight the nuanced shifts in the Chinese government's approach to internal migration, influenced by socioeconomic developments and policy adjustments. Results are visualized through various graphs and discussed comprehensively in the sections below.

### Graphs and Visualizations
![Sentiment Over Time](/path/to/sentiment_graph.png)
*Sentiment analysis of The People's Daily articles from 1980 to 2023.*

![Topic Frequency](/path/to/topic_graph.png)
*Frequency and changes of major topics identified through LDA.*

## Additional Resources
For further details on the methodology and full results, refer to the following links hosted on PlayDavis:
- [Sentiment Analysis Detailed Results](https://linktoplaydavis.com/sentiment_analysis)
- [Topic Modeling Outcomes](https://linktoplaydavis.com/topic_modeling)

## Citing This Work
If you find this study useful in your research, please consider citing it as follows:

