import random
eng_seq=random.sample([0,1,2,3,4,5,6,7,8,9],10)
chi_seq1=random.sample([0,1,2,3,0,1,2,3],8)
chi_seq2=random.sample([0,1,2,3],2)
chi_seq=chi_seq1+chi_seq2
print(eng_seq,chi_seq)
f=open("records_multilingual_exp.txt","r")
curr_epoch_num=int(f.readline())+1
f.close()
f=open("records_multilingual_exp.txt","w")
f.write(str(curr_epoch_num))
f.close()
yaml_files=["las-pyramidal-multilingual-experiment-sequence0.yaml","las-pyramidal-multilingual-experiment-sequence1.yaml","las-pyramidal-multilingual-experiment-sequence2.yaml",\
"las-pyramidal-multilingual-experiment-sequence3.yaml","las-pyramidal-multilingual-experiment-sequence4.yaml","las-pyramidal-multilingual-experiment-sequence5.yaml",\
"las-pyramidal-multilingual-experiment-sequence6.yaml","las-pyramidal-multilingual-experiment-sequence7.yaml","las-pyramidal-experiment-multilingual-sequence8.yaml",\
"las-pyramidal-multilingual-experiment-sequence9.yaml"]
for i in range(len(yaml_files)):
    f=open(yaml_files[i],"r")
    lines=f.readlines()
    f.close()
    f=open(yaml_files[i],"w")
    #print(yaml_files[i])
    j=0
    while j<len(lines):
        curr_line=lines[j].split(':')
        if len(curr_line)<2:
            j+=1
            continue
        if curr_line[1].strip()=="train.tasks.0.src_file":
            f.write(':'.join(curr_line))
            j+=1
            curr_line=lines[j].split(':')
            curr_line[1]=" "+"'{ENG_DATA_DIR}/feat/train"+str(eng_seq[i])+".h5'"+'\n'
        if curr_line[1].strip()=="train.tasks.0.trg_file":
            f.write(':'.join(curr_line))
            j+=1
            curr_line=lines[j].split(':')
            curr_line[1]=" "+"'{ENG_DATA_DIR}/transcript/train"+str(eng_seq[i])+".char'"+'\n'
        if curr_line[1].strip()=="train.tasks.1.src_file":
            f.write(':'.join(curr_line))
            j+=1
            curr_line=lines[j].split(':')
            curr_line[1]=" "+"'{CHI_DATA_DIR}/train"+str(chi_seq[i])+".h5'"+'\n'
        if curr_line[1].strip()=="train.tasks.1.trg_file":
            f.write(':'.join(curr_line))
            j+=1
            curr_line=lines[j].split(':')
            curr_line[1]=" "+"'{CHI_DATA_DIR}/train"+str(chi_seq[i])+".char'"+'\n'
        f.write(':'.join(curr_line))
        j+=1
    f.close()
