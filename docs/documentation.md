## Instructions
Dependencies include but are not limited to:
numpy 1.13
tensorflow 1.4
csv

The project is tested with tensorflow 1.3 and 1.4

The project is still in prototyping phase, so there's close to no support for input.
The main module is the file neural_poet.py. You can train the network by running the module with "training_mode" variable set to "True" on line 66. You can start generating lyrics to stdout by setting the variable to "False".

### Output during training

During training, the net will output examples of probabilities associated with some random word vectors.
Examples of output [here](examples.md)

## Net structure

![Model](model.png?raw=true "model")

## Word vector differences

### Comparison of a generated poetry to actual poems

---
![Plot3](plot3.png?raw=true "Plotted poem")

---
Comparison of a generated poem to actual poem
![Plot1](plot1.png?raw=true "Plotted poem")
![Plot2](plot2.png?raw=true "Plotted poem")
