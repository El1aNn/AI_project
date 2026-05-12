# Project 3 Misclassification Comparison

## 3 documents misclassified in Task A but correct in Task B

- Index 5: 本科未录取还有这些路可以走
  True label: 教育; Task A predicted 房产; Task B predicted 教育.
  Brief explanation: word segmentation can preserve multi-character keywords, names, and domain terms, so Task B may capture the news topic more clearly than isolated characters.
- Index 7: 去新西兰体验舌尖上的饕餮之旅(组图)
  True label: 教育; Task A predicted 房产; Task B predicted 教育.
  Brief explanation: word segmentation can preserve multi-character keywords, names, and domain terms, so Task B may capture the news topic more clearly than isolated characters.
- Index 28: 调查显示：29.5%的人不满意当年所选高考专业
  True label: 教育; Task A predicted 股票; Task B predicted 教育.
  Brief explanation: word segmentation can preserve multi-character keywords, names, and domain terms, so Task B may capture the news topic more clearly than isolated characters.

## 3 documents misclassified in Task B but correct in Task A

- Index 12: 研究生办替考网站续：幕后老板年赚近百万(图)
  True label: 教育; Task A predicted 教育; Task B predicted 社会.
  Brief explanation: character-level input is robust to rare words, short titles, and segmentation mistakes, so Task A can sometimes avoid errors introduced by word tokenization.
- Index 15: 公共英语(PETS)写作中常见的逻辑词汇汇总
  True label: 教育; Task A predicted 教育; Task B predicted 游戏.
  Brief explanation: character-level input is robust to rare words, short titles, and segmentation mistakes, so Task A can sometimes avoid errors introduced by word tokenization.
- Index 17: 九成外国人愿继续在日生活 六成留学生未返校
  True label: 教育; Task A predicted 教育; Task B predicted 房产.
  Brief explanation: character-level input is robust to rare words, short titles, and segmentation mistakes, so Task A can sometimes avoid errors introduced by word tokenization.
