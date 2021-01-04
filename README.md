# Option_Pricing_Python
I use Python to try all the experiments on the classic text book **&lt;Options, Futures and other Derivatives, 8th edition.>**, focusing on the **Black-Scholes Model** and the sensitivity analysis on Greek Letters. The Charpter: The Greek Letters, Page 377-396. All the results and procedures have been tested and compared with the textbook. You can feel safe to use it. If you are the **new beginer for this course** or the **teacher who wants to find well writen Option Prcing code in Python Version**, I think this project will help you. **Please give me a star or folk if you like it.**

### Structure
There are two parts in this project:
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Code.JPG)

**The first part** is that I use the basic data, to calcualte the call option price and put option price. And I also calculate the greek letters, do basic sensitivity analysis on this greek letters. All the results have been checked and compared with the text book. 

**The second part** is that I use the real example data from a stock company, this portfolio is only part of one day's data, there is no any confidential info, thus, it's okay to use. I use this real example data to calculate the greek letters value in the real suitation and do pressure test on the greek letters. (Change the S0 and Sigma, this is one of the most important things the traders care about.)

**The common parameter set used in the text book is shown as below:**
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Basic_data.JPG)

**The real example data is shown as below:**
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Real_example_data.JPG)

### Some Experiment Results
You can find some experiment results in the folder called './fig/', please have a look. If you want to plot other pictures, you can change the code in this project. I have written nearly 500 lines, and I think there will be some materials that are beneficial to you. 
![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Gamma_to_T.png)

![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Theta_to_T.png)

![Image text](https://github.com/Neural-Finance/Option_Pricing_Python/blob/main/fig/Pressure_test.png)
