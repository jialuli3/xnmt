from pypinyin import pinyin,lazy_pinyin,Style
import dynet as dy
import numpy as np
vocab_file="/home/jialu/xnmt/recipes/las-tedlium/chineseVocab.char"

def convert_dictionary(vocab,heter):
    """
    style: 0-no tone, 1 - tone
    """
    vocab_dict={}
    vocab_list=[]
    f=open(vocab,"r",encoding="GBK")
    vocabs = f.readlines()
    for v in vocabs:
        v=v.strip()
        if pinyin(v,heteronym=heter) == []:
            vocab_dict[v]=' '
        else:
            vocab_dict[v]=pinyin(v,heteronym=heter)[0]
        vocab_list.append(v)
    f.close()
    print(len(set(vocab_list)))

#convert_dictionary(vocab_file,False)
# define the parameters
m = dy.ParameterCollection()
W = m.add_parameters((8,2),name="W")
V = m.add_parameters((1,8))
b = m.add_parameters((8))
para_list=m.parameters_list()

# renew the computation graph
dy.renew_cg()

# create the network
x = dy.vecInput(2) # an input vector of size 2.
output = dy.logistic(V*(dy.tanh((W*x)+b)))
# define the loss with respect to an output y.
y = dy.scalarInput(0) # this will hold the correct answer
loss = dy.binary_log_loss(output, y)

# create training instances
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()

# train the network
trainer = dy.SimpleSGDTrainer(m)

total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    total_loss += loss.value()
    W1_prev=dy.parameter(W)
    print(W1_prev.value())
    #prev_w=(m.parameters_list()[0].value())
    loss.backward()
    trainer.update()
    #print(prev_w-m.parameters_list()[0].value())
    if (seen_instances > 1 and seen_instances % 100 == 0):
        print("average loss is:",total_loss / seen_instances)
