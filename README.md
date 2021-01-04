# Option_Pricing_Python
I use Python to try all the experiments on the classic text book **&lt;Options, Futures and other Derivatives, 8th edition.>**, focusing on the **Black-Scholes Model** and the sensitivity analysis on Greek Letters. The Charpter: The Greek Letters, Page 377-396. All the results and procedures have been tested and compared with the textbook. You can feel safe to use it. If you are the **new beginer for this course** or the **teacher who wants to find well writen Option Prcing code in Python Version**, I think this project will help you. Please give me a star or folk if you like it. 

### Structure
There are two parts in this project:
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Code.JPG)

The first part is that I use the basic data, to calcualte the call option price and put option price. And I also calculate the greek letters, do basic sensitivity analysis on this greek letters. All the results have been checked and compared with the text book. 

The second part is that I use the real example data from a stock company, this portfolio is only part of one day's data, there is no any confidential info, thus, it's okay to use. I use this real example data to calculate the greek letters value in the real suitation and do presure test on the greek letters. (Change the S0 and Sigma, this is one of the most important things the traders care about.)

The common parameter set used in the text book is shown as below:
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Basic_data.JPG)

The real example data is shown as below:
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Real_example_data.JPG)

### Example
If we should rank() the price of stocks in the sample trading day. In order to let the neural network learn this operator, we have to let sample1 see the features belong to sample2. As metioned above, the traditional structure can't see the features belong to other sample. Thus, we should put all sample's data into one picture, and let it serves as X. The output should be all sample's value, which is Pred_Y. And the real factor value array should be Y. The mean squared error of Y and pred_Y, severs as loss function.


### Project Structure
```
Main.py --You can run it to train the network and test the data.
```

```
Data_processing.py --Built the figure data
```

```
Querry.py --a sub function for Data_processing, here, you can put in a formula, which produces x and y
```

```
Lenet.py --The neural network model structure (Tensorflow 1.3)
```

```
hyper_parameters.py --all the hyper-parameters
```

### Input X and Output Y
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/1.png)
![Image text](https://github.com/Neural-Finance/Cross_sample_financial_feature_engineering/blob/master/fig/2.png)

**Lenet (A kind of CNN network structure)**

I want to mention that, sometimes, 1*m kernels will be more helpful in this task, because it's more like a time series operation. And the second point is that, I know GNN is very hot nowadays, however, you should know the relationship between different stocks. If you are pretty sure about it, then you should use GNN here. If you want to freely learn their relationship, both in time series or cross section relationship, then the tradtional CNN will be a better choice.
