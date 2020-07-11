Sequential Learning Paper

|  Year | Category  | Title  |  
|---|---|---|
| NIPS2014	  | Seq2Seq with RNN  | [Sequence to Sequence Learning with Neural Networks](#nipsd2014)  |
| ICLR2015  | Attention and bi-RNN  | [Neural Machine Translation by Jointly Learning to Align and Translate](#iclr2015)  |
| ACL2016  | Self-attention with LSTM  | [Long Short-Term Memory-Networks for Machine Reading](#acl2016)  | 
| NIPS2017  | Transformer with multi-head attention | [Attention Is All You Need](#nips2017)  |



## <a id="nips2014">[Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Proposed neural machine translation and proposed the encoding/decoding architecture. The input first go through an encoder (Multilayer LSTM) to encode the information into a fixed length memory, then the information in the fixed length memory are feed to the decoder to get the output. Additionally, the model was feed with input sentence in the reverse order. The argument is that the reverse order can make it easy for SGD to "establish communication" between the input and the output. **This could be true for later words in the input sentence, but the information of the earlier words will loss even more, as they are further from the targets. However, the overall benefits are larger than the drawbacks.**

<p align="center">
    <img src="imgs/nips2014.png">
</p>


## <a id="iclr2015">[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

Address the limitation of fixed length memory by extend the memory storage with an attention network (a feed forward net with 1 hidden layer). Additionally, both encoder/decoder are using bi-directional RNN to provide both the pre-context and post-context of the current target.

The input of the attention network is the concatenation of the previous hidden state of the decoder $h_{i-1}$ and the a hidden state $h_j$, the output would be the weight/importance of the input $x_j$ to $y_i$. Note that all the input have some contribution to the target $y_i$. The softmax will be applied to all the weight for normalization and the weighted sum of the hidden states $h_1, ..., h_t$ with respect with the weights will be the context vector $c_i$ for target $y_i$. Then a nonlinear, potentially multi-layered function $g(y_{i-1}, s_i, c_i)$ is used to get the target output.

<p align="center">
    <img align="center" src="imgs/iclr2015.png">
</p>




## <a id="acl2016">[Long Short-Term Memory-Networks for Machine Reading](https://arxiv.org/pdf/1601.06733.pdf)

Addressed the limitation of recurrent NNs: 1) EliminatedGradient vanish/exploding; 2) Unable to keep long term memory for earlier input in the sequence; 3) No reasoning over structure and no explicit relationship information retained among input tokens.

Improved the LSTM by replacing the gates (forget, input, output) with networks.


## <a id="nips2017">[Attention Is All You Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

Abandon recurrent NN and only use feed forward and attention mechanism for feature extraction for sequential data.


