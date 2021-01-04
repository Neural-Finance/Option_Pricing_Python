# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:47:49 2020

@author: Alex Fang
"""
import numpy as np 
import pandas as pd
from math import log, sqrt, exp, pi
from scipy.stats import norm
import matplotlib.pyplot as plt

#Reference market data: Book, Options, Futures, and Other Derivatives, 8th edition,
#The Charpter: The Greek Letters, Page 377-396
S0=49
K=50
T=0.3846 #20weeks, unit: year
r=0.05 #unit: year
sigma=0.20

#Class Option Pricing
class Option_Price():
    
    def init(self):
        pass
    
    #Function set 1, B-S model
    def BSM(self,S0, K, T, r, sigma, type_):
        
        d1 = (log(S0/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T)
        d2 = d1 - sigma * sqrt(T)
        Callprice = S0 * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
        Putprice = -S0 * norm.cdf(-d1) + K*exp(-r*T)* norm.cdf(-d2)
        if type_=="call" or type_=="Call":
            print("This Call Option Price is: %.4f" % Callprice)
        elif type_=="put" or type_=="Put" :
            print("This Put Option Price is: %.4f" % Putprice)
        else:
            print("No match, check your type...")


    #Function set 2, sensitive analysis on its important variables.
    #Function set 3, some visualization tools
    def Delta_to_S(self,S0, K, r, sigma):

        Call=[]
        Put=[]
        for S in range(30,80):
            Call.append(norm.cdf((log(S/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T)))
            Put.append(norm.cdf((log(S/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T))-1)
   
        plt.plot(range(len(Put)),Put,label='Put')
        plt.plot(range(len(Call)),Call,label='Call')
        plt.xlabel('S0')
        plt.ylabel('Delta')
        plt.legend(loc='upper right')
        plt.savefig("./fig/Delta_to_S.png")
        plt.close()
        
    
    def Delta_to_T(self,S0, K, r, sigma):

        In_the_time=[]
        At_the_time=[]
        Out_of_time=[]
        for T1 in range(1,200):
            T1/=100
            In_the_time.append(norm.cdf((log(S0/(S0-10)) + (r + 0.5 * sigma**2) * T1) / sigma / sqrt(T1)))
            At_the_time.append(norm.cdf((log(S0/(S0)) + (r + 0.5 * sigma**2) * T1) / sigma / sqrt(T1)))
            Out_of_time.append(norm.cdf((log(S0/(S0+10)) + (r + 0.5 * sigma**2) * T1) / sigma / sqrt(T1)))
        
        plt.plot(range(len(In_the_time)),In_the_time,label='In_the_time')
        plt.plot(range(len(In_the_time)),At_the_time,label='At_the_time')
        plt.plot(range(len(In_the_time)),Out_of_time,label='Out_of_time')
        plt.xlabel('T')
        plt.ylabel('Delta')
        plt.legend(loc='upper right')
        plt.savefig("./fig/Delta_to_T.png")
        plt.close()
    

    def Theta(self,S0, K,T, r, sigma, type_,plot=False):
        
        d1 = (log(S0/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T)
        d2 = d1 - sigma * sqrt(T)
        def n_1(x):
            return 1/sqrt(2*pi)*exp(-x**2/2)
            
        
        Call=-S0*n_1(d1)*sigma/2/sqrt(T)-r*K*exp(-r*T)*norm.cdf(d2)
        Put=-S0*n_1(d1)*sigma/2/sqrt(T)+r*K*exp(-r*T)*norm.cdf(-d2)
        
        if plot:
            return Call
        
        if type_=="call" or type_=="Call":
            print("This Call Option's theta is : %.4f" % Call)
        elif type_=="put" or type_=="Put" :
            print("This Put Option's theta is : %.4f" % Put)
        else:
            print("No match, check your type...")
        
        
        
    def Theta_to_S(self,S0, K,T, r, sigma):
        
        Call=[]
        x=[]
        for S in range(30,80):
            x.append(S)
            Call.append(self.Theta(S, K, T, r, sigma, 'call',plot=True))
            
        plt.plot(x,Call,label='Call: Theta')
        plt.plot([K]*10,np.array(range(1,101,10))/-10,'-.',label='K')
        plt.xlabel('S0')
        plt.ylabel('Theta')
        plt.legend(loc='upper right')
        plt.savefig("./fig/Theta_to_S(.png")
        plt.close()
        
    
    def Theta_to_T(self,S0, K,T, r, sigma):

        In_the_time=[]
        At_the_time=[]
        Out_of_time=[]
        for T1 in range(1,100):
            T1/=100
            In_the_time.append(self.Theta(S0, S0-2, T1, r, sigma, 'call',plot=True))
            At_the_time.append(self.Theta(S0, S0, T1, r, sigma, 'call',plot=True))
            Out_of_time.append(self.Theta(S0, S0+2, T1, r, sigma, 'call',plot=True))
        
        plt.plot(range(len(In_the_time)),In_the_time,label='In_the_time: Theta')
        plt.plot(range(len(In_the_time)),At_the_time,label='At_the_time: Theta')
        plt.plot(range(len(In_the_time)),Out_of_time,label='Out_of_time: Theta')
        plt.xlabel('T')
        plt.ylabel('Theta')
        plt.legend(loc='upper right')
        plt.savefig("./fig/Theta_to_T.png")
        plt.close()
    
    
    def Gamma(self,S0, K,T, r, sigma, type_,plot=False):
        
        d1 = (log(S0/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T)
        d2 = d1 - sigma * sqrt(T)
        def n_1(x):
            return 1/sqrt(2*pi)*exp(-x**2/2)
            
        Call=n_1(d1)/S0/sigma/sqrt(T)
        Put=-S0*n_1(d1)*sigma/2/sqrt(T)+r*K*exp(-r*T)*norm.cdf(-d2)
        
        if plot:
            return Call
        
        if type_=="call" or type_=="Call":
            print("This Call Option's gamma is : %.4f" % Call)
        elif type_=="put" or type_=="Put" :
            print("This Put Option's gamma is : %.4f" % Put)
        else:
            print("No match, check your type...")
    
    
    def Gamma_to_T(self,S0, K,T, r, sigma):

        In_the_time=[]
        At_the_time=[]
        Out_of_time=[]
        for T1 in range(1,100):
            T1/=100
            In_the_time.append(self.Gamma(S0, S0-2, T1, r, sigma, 'call',plot=True))
            At_the_time.append(self.Gamma(S0, S0, T1, r, sigma, 'call',plot=True))
            Out_of_time.append(self.Gamma(S0, S0+2, T1, r, sigma, 'call',plot=True))
        
        plt.plot(range(len(In_the_time)),In_the_time,label='In_the_time: Gamma')
        plt.plot(range(len(In_the_time)),At_the_time,label='At_the_time: Gamma')
        plt.plot(range(len(In_the_time)),Out_of_time,label='Out_of_time: Gamma')
        plt.xlabel('T')
        plt.ylabel('Gamma')
        plt.legend(loc='upper right')
        plt.savefig("./fig/Gamma_to_T.png")
        plt.close()
        
        
    def Vega(self,S0, K,T, r, sigma, type_,plot=False):
        
        d1 = (log(S0/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T)
        d2 = d1 - sigma * sqrt(T)
        def n_1(x):
            return 1/sqrt(2*pi)*exp(-x**2/2)
            
        Call=S0*sqrt(T)*n_1(d1)
        Put=S0*sqrt(T)*n_1(d1)
        
        if plot:
            return Call
        
        if type_=="call" or type_=="Call":
            print("This Call Option's Vega is : %.4f" % Call)
        elif type_=="put" or type_=="Put" :
            print("This Put Option's Vega is : %.4f" % Put)
        else:
            print("No match, check your type...")
    
    
    def Vega_to_S(self,S0, K,T, r, sigma):
        
        Call=[]
        x=[]
        for S in range(30,80):
            x.append(S)
            Call.append(self.Vega(S, K, T, r, sigma, 'call',plot=True))
            
        plt.plot(x,Call,label='Call: Vega')
        plt.plot([K]*10,np.array(range(1,201,20))/10,'-.',label='K')
        plt.xlabel('S0')
        plt.ylabel('Vega')
        plt.legend(loc='upper right')
        plt.savefig("./fig/Vega_to_S.png")
        plt.close()


    def Rho(self,S0, K,T, r, sigma, type_,plot=False):
        
        d1 = (log(S0/K) + (r + 0.5 * sigma**2) * T) / sigma / sqrt(T)
        d2 = d1 - sigma * sqrt(T)
        def n_1(x):
            return 1/sqrt(2*pi)*exp(-x**2/2)
            
        
        Call=K*T*exp(-r*T)*norm.cdf(d2)
        Put=-K*T*exp(-r*T)*norm.cdf(-d2)
        
        if plot:
            return Call
        
        if type_=="call" or type_=="Call":
            print("This Call Option's Rho is : %.4f" % Call)
        elif type_=="put" or type_=="Put" :
            print("This Put Option's Rho is : %.4f" % Put)
        else:
            print("No match, check your type...")
    
    
    def Real_example(self):
        
        #calculate the table in excel
        data=pd.read_excel("./Portfolio.xlsx").iloc[1:]
        data=data[data['Type']=='Put']
        data.index=range(len(data))
        S0=4.572
        t0='2020-09-30'
        from datetime import datetime
        input_columns=['S0','T','r','K','sigma']
        input_data=np.zeros((len(data),5))
        input_data[:,0]=np.array([S0]*len(data)).reshape(-1,)
        temp=[]
        for i in range(len(data)):
            temp.append(np.abs((data['Maturity'][i]-datetime.strptime(t0, '%Y-%m-%d')).days)/252)
        
        input_data[:,1]=np.array(temp).reshape(-1,)
        input_data[:,2]=np.array(data['Rate']-data['SSRate']).reshape(-1,)
        input_data[:,3]=np.array(data['Strike']).reshape(-1,)
        input_data[:,4]=np.array(data['ActVol']).reshape(-1,)
        
        d1 = (np.log(input_data[:,0]/input_data[:,3]) + (input_data[:,2] + 0.5 * input_data[:,4]**2) * input_data[:,1]) / input_data[:,4] / np.sqrt(input_data[:,1])
        d2 = d1 - input_data[:,4] * np.sqrt(input_data[:,1])
        delta = norm.cdf(d1)-1
        delta=delta*data['NetPos']*data['Multi']
        
        cash_delta=delta*S0
        
        def n_1(x):
            return 1/np.sqrt(2*pi)*np.exp(-np.square(x)/2)

        vega=input_data[:,0]*np.sqrt(input_data[:,1])*n_1(d1)
        vega*=data['NetPos']
        vega*=data['Multi']
        
        gamma=input_data[:,0]*n_1(d1)*input_data[:,4]/2/np.sqrt(input_data[:,1])-input_data[:,2]*input_data[:,3]*np.exp(-input_data[:,2]*input_data[:,1])*norm.cdf(-d2)
        gamma*=data['NetPos']
        gamma*=data['Multi']
        
        new_d1=(np.log(input_data[:,0]*1.01/input_data[:,3]) + (input_data[:,2] + 0.5 * input_data[:,4]**2) * input_data[:,1]) / input_data[:,4] / np.sqrt(input_data[:,1])
        new_cash_delta = (norm.cdf(new_d1)-1)*data['NetPos']*data['Multi']*S0
        #the change rate of 0.01S
        cash_gamma=gamma*np.sqrt(input_data[:,0])/100
        
        theta=-input_data[:,0]*n_1(d1)*input_data[:,4]/2/np.sqrt(input_data[:,1])+input_data[:,2]*input_data[:,3]*np.exp(-input_data[:,2]*input_data[:,1])*norm.cdf(-d2)
        theta=theta*data['Multi']*data['NetPos']/252

        summary=pd.DataFrame()
        summary['Delta']=delta
        summary['CashDelta']=cash_delta
        summary['Vega']=vega/100
        summary['Gamma']=gamma
        summary['CashGamma']=cash_gamma
        summary['Theta']=theta
        return summary


    def Real_example2(self):
        
        #calculate the change of greek letters when S0 and Volatility changes
        raw=pd.read_excel("./Portfolio.xlsx")
        raw.index=range(len(raw))
        S0=4.572
        t0='2020-09-30'
        from datetime import datetime
        
        def calculate_greek(a,b,type_):
            
            data=raw[raw['Type']==type_]
            data.index=range(len(data))
            input_columns=['S0','T','r','K','sigma']
            input_data=np.zeros((len(data),5))
            input_data[:,0]=S0
            #change the underlying assets S0
            input_data[:,0]*=a
            temp=[]
            for i in range(len(data)):
                temp.append(np.abs((data['Maturity'][i]-datetime.strptime(t0, '%Y-%m-%d')).days)/252)
            
            input_data[:,1]=np.array(temp).reshape(-1,)
            input_data[:,2]=np.array(data['Rate']-data['SSRate']).reshape(-1,)
            input_data[:,3]=np.array(data['Strike']).reshape(-1,)
            input_data[:,4]=np.array(data['ActVol']).reshape(-1,)
            #change the underlying assets S0
            input_data[:,4]+=b
            
            d1 = (np.log(input_data[:,0]/input_data[:,3]) + (input_data[:,2] + 0.5 * input_data[:,4]**2) * input_data[:,1]) / input_data[:,4] / np.sqrt(input_data[:,1])
            d2 = d1 - input_data[:,4] * np.sqrt(input_data[:,1])
            if type_=='Put':
                delta = norm.cdf(d1)-1
            elif type_=='Call':
                delta = norm.cdf(d1)
            
            delta=delta*data['NetPos']*data['Multi']
            
            cash_delta=delta*S0
            
            def n_1(x):
                return 1/np.sqrt(2*pi)*np.exp(-np.square(x)/2)
    
            vega=input_data[:,0]*np.sqrt(input_data[:,1])*n_1(d1)*data['NetPos']*data['Multi']
            gamma=n_1(d1)/input_data[:,0]/input_data[:,4]/np.sqrt(input_data[:,1])
            gamma=gamma*data['NetPos']*data['Multi']
            
            '''
            new_d1=(np.log(input_data[:,0]*1.01/input_data[:,3]) + (input_data[:,2] + 0.5 * input_data[:,4]**2) * input_data[:,1]) / input_data[:,4] / np.sqrt(input_data[:,1])
            if type_=='Put':
                new_cash_delta = (norm.cdf(new_d1)-1)*data['NetPos']*data['Multi']*S0*1.01
            elif type_=='Call':
                new_cash_delta = norm.cdf(new_d1)*data['NetPos']*data['Multi']*S0*1.01
            
            #the change rate of 0.01S
            cash_gamma=new_cash_delta-cash_delta
            '''
            cash_gamma=gamma*np.square(input_data[:,0])/100
            
            if type_=='Put':
                theta=-input_data[:,0]*n_1(d1)*input_data[:,4]/2/np.sqrt(input_data[:,1])+input_data[:,2]*input_data[:,3]*np.exp(-input_data[:,2]*input_data[:,1])*norm.cdf(-d2)
            elif type_=='Call':
                theta=-input_data[:,0]*n_1(d1)*input_data[:,4]/2/np.sqrt(input_data[:,1])-input_data[:,2]*input_data[:,3]*np.exp(-input_data[:,2]*input_data[:,1])*norm.cdf(d2)
            
            theta=theta/252*data['NetPos']*data['Multi']
            summary=pd.DataFrame()
            summary['Delta']=delta
            summary['CashDelta']=cash_delta
            summary['Vega']=vega/100
            summary['Gamma']=gamma
            summary['CashGamma']=cash_gamma
            summary['Theta']=theta
            
            return [np.sum(summary['CashDelta']),np.sum(summary['CashGamma']),np.sum(summary['Theta']),np.sum(summary['Vega'])]
        

        m1,m2,m3,m4,ind_=[],[],[],[],[]
        for a in [1,1.05,1.1,1.2,0.95,0.9,0.8]:
            for b in [0.1,0.2,-0.1,-0.2]:
                ans=np.array(calculate_greek(a,b,'Put'))+np.array(calculate_greek(a,b,'Call'))
                x1=(ans[0]+raw['CashDelta'][0])/10000
                x2=(ans[1]+raw['CashGamma'][0])/10000
                x3=(ans[2]+raw['Theta'][0])/10000
                x4=(ans[3]+raw['Vega'][0])/10000
                ind="SO changes "+"%.2f"%(a-1)+", Vol changes "+str(b)
                ind_.append(ind)
                m1.append(x1)
                m2.append(x2)
                m3.append(x3)
                m4.append(x4)
        
        
        res=pd.DataFrame()
        res['Sensitivity']=ind_
        res['CashDelta']=m1
        res['CashGamma']=m2
        res['Theta']=m3
        res['Vega']=m4

        return res
        

#Just use BS model to calculate the option price
Calculator=Option_Price()
Calculator.BSM(S0, K, T, r, sigma, 'call')
#Calculate the Greek Letters, and plot all the pictures shown in the text book
#For the charpter, the greek letters.
Calculator.Delta_to_S(S0, K, r, sigma)
Calculator.Delta_to_T(S0, K, r, sigma)
Calculator.Theta(S0, K, T, r, sigma,type_='Call')
Calculator.Theta_to_S(S0, K, T, r, sigma)
Calculator.Theta_to_T(S0, K, T, r, sigma)                         
Calculator.Gamma(S0, K, T, r, sigma,type_='Call')
Calculator.Gamma_to_T(S0, K, T, r, sigma)
Calculator.Vega(S0, K, T, r, sigma,type_='Call')
Calculator.Vega_to_S(S0, K, T, r, sigma)
Calculator.Rho(S0, K, T, r, sigma,type_='Call')
#Use BS model to calculate Greek Letters
#The data is based on a portfolio, given by a stock company.
summary=Calculator.Real_example()
#Use BS model to calculate Greek Letters, and do pressure test
#Test it on different underlying assets price S0, and volatility sigma.
summary=Calculator.Real_example2()            
            
            
            
            
            
            
    
        
        