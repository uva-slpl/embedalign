Implementation of [Deep Generative Model for Joint Alignment and Word Representation](https://arxiv.org/abs/1802.05883).

EmbedAlign represents words in context using a deep generative model that learns from bilingual parallel corpora. 
In a nutshell, it generates two strings (one in L1, one in L2) one word at a time. It generates L1 first by sampling word representations from a Gaussian prior, 
then it generates L2 by aligning it to L1 using an IBM2-type model. Crucially, L2 words are generated from the stochastic latent representation of L1 words. 
This essentially has the effect of biasing learned representations towards capturing synonymy (under the assumption that translation data make up for a form of
noisy sense annotation). 


# Dependencies

* `python 3.5`
* `tensorflow r1.4`

        pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp35-cp35m-linux_x86_64.whl

* [dgm4nlp](https://github.com/uva-slpl/dgm4nlp) tag `embedalign.naacl2018.submission`

# Example

Check `test/hansards.py` for an example. 
