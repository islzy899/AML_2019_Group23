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
### 3.1 Experiments with different step sizes (0.01,0.001,0.0001) ,the number of iteration chosen is 10000
