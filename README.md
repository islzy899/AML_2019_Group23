# AML_2019_Group23
## 1. Explain concisely, in layman’s terms without using any formulae
In machine learning, optimization is a important part.Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).In addtion, gradient descent is best used in machine learing when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.
Vanilla gradient descent is a type of gradient descent which processes all the training examples for each iteration of gradient descent. For every iteration, the given point θ will be updated by subtracting the gradient(θ) multiplied by the step size (η) until the lastest gradient(θ) less than the torlerance that we chose before. And each iteration will carried out in the right direction and take the steepest step at the current position. Finally we can get the minimum/maximum value of the function.
Though vanilla gradient descent is a useful way in machine learning, it has some drawbacks. For example,some parameter will oscillate because of step size(η) and may stuck at the saddle point. Hence, there has two modifications to plain vanilla gradient descent: momentum, Nesterov’s Accelerated Gradient (NAG). Momentum can accelerate the speed of the dimension where the gradient direction remains unchanged and slow the updating speed of the dimension where the gradient direction changes, which can accelerate convergence and reduce oscillation. Nesterov’s Accelerated Gradient (NAG) is an extended version of the momentum method. There is a correction factor has been added to the standard momentum method so the NAG perform better convergence than momentum method.
## 2.Chose a function : Three-Hump-Camel function
![3d](https://user-images.githubusercontent.com/52762661/61016257-ea199100-a386-11e9-8840-05d2fe439a3a.png)

The function is usually evaluated on the square xi ∈ [-4, 4], for all i = 1, 2. 

The minimum of the function is f(0,0)=0.
## 3. Plain vanilla Gradient Descent: starting point(x1=3,x2=3)
#### 3.1 Experiments with different step sizes (0.01,0.001,0.0001) ,the number of iteration chosen is 10000
![step111](https://user-images.githubusercontent.com/52762661/61017332-b93b5b00-a38a-11e9-999c-069a2c6c695e.png)
![step1](https://user-images.githubusercontent.com/52762661/61017037-b5f39f80-a389-11e9-968c-cc69c0393dc5.png)
![step2](https://user-images.githubusercontent.com/52762661/61017068-cd328d00-a389-11e9-82f8-89b7e7651555.png)
[figure1-6]The 6 figures show the path to find the minimum value of the function in different step size of gradient descent.
Compare to this 6 figures, we can see that the smaller step size, the longer time to computer and the larger of iteration number.

![v](https://user-images.githubusercontent.com/52762661/61021710-8ac57c00-a39a-11e9-9cb0-862da87fe950.png)

[figure7]This figure plot the step size with corresponding minimun function value. As the figure show, if the step size smaller than 0.001, then the minimum value of function will be larger than the actual minimum value(0). In other words, if the step size is too small, the gradient descent process can't find the actual minimum value of the function.

![v2](https://user-images.githubusercontent.com/52762661/61021711-8b5e1280-a39a-11e9-9b0c-772dbae32d03.png)

[figure8]This figure plot the step size with coreesponding number of iteration. From the figure, we can see that as the step size becomes smaller, the number of iteration will become larger, which match the information shown in figure1 to figure6.

## 4. Momentum Gradient Descent: starting point(x1=3,x2=3) 
#### 4.1 Experiments with different step sizes (0.01,0.001,0.0001) ,the number of iteration chosen is 10000
![mo](https://user-images.githubusercontent.com/52762661/61021679-69fd2680-a39a-11e9-9877-d10ff983a68a.png)

[figure9]This figure plot the step size with corresponding minimum value of fucntion. The figure shows the similar result as vanilla gradient descent figure7. The result is stable when step size bigger than 0.001.

## 5. Nesterov’s Accelerated Gradient (NAG): starting point(x1=3,x2=3) 
#### 5.1 Experiments with different step sizes (0.01,0.001,0.0001) ,the number of iteration chosen is 10000
![nag](https://user-images.githubusercontent.com/52762661/61021683-6bc6ea00-a39a-11e9-8f56-ba79a0693f6e.png)

[figure10]This figure plot the step size with corresponding minimum value of function. As the figure shows, the minimum value has a steep decrease as step size changes from 0 to 0.001. After 0.001,the value decreases a bit slowly as step size is between 0.001 to 0.01.

## 6. Conclusion
Plain vanilla gradient descent is a useful and simple way to find the minimum value of a function. However, it has some drawbacks such as too much time consuming, effected easily by some parameters. Compared with the momentum and Nesterov’s Accelerated gradient descend, this two model perfrom a better result in finding the minimum more quickly in smaller iteration times and find a accurate minimum value of the function.
