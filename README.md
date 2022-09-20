# Автоматическое выявление токсичных комментариев
![comments_nlp](https://st4.depositphotos.com/2454953/21024/i/600/depositphotos_210245966-stock-photo-ignore-haters-word-cloud-hand.jpg)
## Описание проекта

**ЦЕЛЬ ПРОЕКТА:**

Для `интернет-магазина` построить **модель классификации** комментариев на негативные и позитивные.

Каждый комментарий - это обратная связь и предложения правок к редактируемым описаниям товаров, которые формируются самими пользователями (свое вики-сообщество).

Модель предсказаний позволит эффективно **модерировать контент** и отслеживать `токсичные комментарии`. 

Целевой KPI - метрика `F1-score <= 0.75` 

=========================================================================================

**ЛИЧНЫЕ ЦЕЛИ:**


Применить на практике технику **TF-IDF** (анализ значимости слов в текстах) и провести небольшое исследование, чтобы узнать, какие `подходы можно использовать` для аналитики и улучшения предсказаний.

В дополнение:

+ вспомнить основы **регулярных** выражений


+ испытать готовую библиотеку для борьбы с дисбалансом `imbLearn` (вместо самописных функций)


+ научиться использовать метод Кси-Квадрат (`chi-square`) для анализа **корреляции категорий**

[Посмотреть проект](Automatic_detection_of_toxic_comments_v1.ipynb)

## Навыки

<div class="alert alert-success">
<br> ✔️ Исследовательский анализ </br>
<br> ✔️ NLP           ✔️ TF-IDF            ✔️ Токенизация</br>
<br> ✔️ Классификация ✔️ Регуляризация</br>
<br> ✔️ Стемминг      ✔️ Лемматизация      ✔️ Частотный анализ</br>
</div>

## Исходные данные


```python
import pandas as pd

pd.set_option("display.max_colwidth", 500)
df = pd.read_csv("/datasets/toxic_comments.csv")

df.sample(5)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44661</th>
      <td>There is only one way to settle this dilemma, we vote for the color of the infobox, Oakland Athletics or New York Yankees.  Having A's colors on Reggie Jackson's page is the most worst thing ever.  Everyone remembers Reggie as a Yankee, the three homeruns in the world series, the argument with Billy Martin, etc.  Reggie had his greatest success as a Yankee.  And lets face it nobody likes the Oakland A's #9 Reggie Jackson but everbody loves the New York Yankees #44 Reggie Jackson.  Reggie wil...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>52006</th>
      <td>Comment on the question about including separate sections for health, environmental, and economic damage.  I don't feel that it is appropriate for this article about BP to include these sections.  I believe that the editors that have argued for including these sections have presented their arguments very well, but it could also be argued that including such a brief summary (as must be) in one sense tends to minimize the issues.  But mainly, the information just seems out of place in this art...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9770</th>
      <td>Reworked to something verifiable.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>127777</th>
      <td>disruption \n\ni am not disrupting it YOU ARE \n\nI have to warn that Richhno to stay away from my project, Vanity Kills. \n\nPlease do not bother me (or any other team member) with this crap again.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>142387</th>
      <td>"\n\nThe sources given for monarchs seems to give a finite number to them, so no, they do not have infinite wealth. Many monarchs never had their wealth valued, but those that did shoudl be here, this is the lsit for them afterall and they are not knocking anyone off by appearing. [tk]  "</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
