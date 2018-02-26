# Run the evaluation script for $testset set
#  then add a line to this script
#  and you will get a digest 

# encoder: bow, rnn z and rnn s & z
# epochs: 30
# annealing: no, yes, yes-wait

model=embedalign
corpus=europarl.en-de
experiments="${HOME}/experiments"
#criterion=validation.aer
criterion=validation.objective
testset=test.lst
script=../../lst_output_digest.py

# To change this, just change your model path (gen:......) and give your model an alias (the parameter of the script)

# BOW
#cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam\,europarl/*/*/{0,1}/$testset.best.$criterion.results | python3 $script bow-no
#cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam\,anneal\,europarl/*/*/{0,1}/$testset.best.$criterion.results | python3 $script bow-yes | tail -n1

# RNN Z
#cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,europarl/*/*/{0,1}/$testset.best.$criterion.results | python3 $script rnn-no | tail -n1
#cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,anneal\,europarl/*/*/{0,1}/$testset.best.$criterion.results | python3 $script rnn-yes | tail -n1
#cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,wait\,anneal\,europarl/*/*/{0,1}/$testset.best.$criterion.results | python3 $script rnn-wait | tail -n1

# RNN S & Z
#cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-s_rnn-z\,opt\:adam\,e:10\,anneal\,europarl/*/*/{0,1}/$testset.best.$criterion.results | python3 $script rnn-sz-wait | tail -n1

#baseline mono
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam\,europarl\,dummy/*/*/{0,1}/$testset.best.$criterion.results | python3 $script bow-mono
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,europarl\,dummy/*/*/{0,1}/$testset.best.$criterion.results | python3 $script rnn-mono
