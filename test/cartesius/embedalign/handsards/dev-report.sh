# Run the evaluation script for $testset set
#  then add a line to this script
#  and you will get a digest 

# encoder: bow or rnn
# epochs: 30 or 50
# annealing: no, yes, no-KL, yes-wait 

model=embedalign
corpus=handsards.en-fr
experiments="${HOME}/experiments"
criterion=validation.aer
testset=dev
script=../../decoder_output_digest.py

# To change this, just change your model path (gen:......) and give your model an alias (the parameter of the script)

# BOW
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam/*/*/{0,1}/$testset.best.$criterion.results | python $script bow-30-no 
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam\,anneal/*/*/{0,1}/$testset.best.$criterion.results | python $script bow-30-yes | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam\,anneal-wait/*/*/{0,1}/$testset.best.$criterion.results | python $script bow-30-yes-wait | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam\,no-KL/*/*/{0,1}/$testset.best.$criterion.results | python $script bow-30-yes-noKL | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam-50epochs/*/*/{0,1}/$testset.best.$criterion.results | python $script bow-50-no | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:bow-z\,opt\:adam-50epochs\,anneal/*/*/{0,1}/$testset.best.$criterion.results | python $script bow-50-yes | tail -n1

# RNN
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam/*/*/{0,1}/$testset.best.$criterion.results | python $script rnn-30-no | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,anneal/*/*/{0,1}/$testset.best.$criterion.results | python $script rnn-30-yes | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,anneal-wait/*/*/{0,1}/$testset.best.$criterion.results | python $script rnn-30-yes-wait | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam\,no-KL/*/*/{0,1}/$testset.best.$criterion.results | python $script rnn-30-noKL | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam-50epochs/*/*/{0,1}/$testset.best.$criterion.results | python $script rnn-50-no | tail -n1
cat $experiments/$model/$corpus/gen\:128\,inf\:rnn-z\,opt\:adam-50epochs\,anneal/*/*/{0,1}/$testset.best.$criterion.results | python $script rnn-50-yes | tail -n1
