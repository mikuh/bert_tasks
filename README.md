# bert_tasks
利用bert实现各种任务.

任务的框架基于谷歌官方的BERT：https://github.com/google-research/bert

与训练模型根据自己的训练数据情况选择适当的,github上有很多,放在vocab_file目录下.


## 用法
1. clone google 官方的bert模型: `git clone https://github.com/google-research/bert`
2. 将本项目的文件copy到bert目录中
3. 准备自己的数据和与训练模型,位置可以自己指定,具体参数在运行文件中指定,可自行修改.

   注意一点:序列标注模型标注时和训练时请使用相同的分词词典,否则长度会对应不上,中英文都可以使用,还有就是标注数据可能回与
   tokenize后的词对应不上,这个需要注意处理.可以参考官方bert模型中的处理方式.
4. 运行任务代码
- 文本分类

    train: `run_classifier.sh`
    
    test: `test_classifier.sh`
- 序列标注

    train: `run_ner.sh`
    
    test: `test_ner.sh`
    

    
    
