### Changelog

2020/08/14
* Add model dynamic quantization.

2020/08/12
* Add model profiling.

2020/08/02
* Use batch first and use pad_sequence.
* Enable model checkpoint save and load.

2020/07/31
* Enable bi-direction and multi-layer RNN.

2020/07/28
* Deprecate TextCNNConfig and RNNConfig by making Config flexible to accept arbitrary params.

2020/07/26
* Allow config to set arbitrary values

2020/07/25
* Make RNN classification model configurable to use RNN, LSTM, and GRU
* Make RNN classification model configurable on the number of layers

2020/07/24
* Refactored the file structure by separating configs, models, and training logic

2020/07/23
* Fix the bug that the weights conv layers are not correctly updated during back-propagation

2020/07/22
* Make TextCNN graph configurable on number of conv branches