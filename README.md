# Logistic regression

## Part I

1. Implement a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function with a signature `func sigmoid(z float64) float64`.
   Add unit tests for the function. Verify the values for `z = -1`, `z = 0`, `z = 1`, `z = 2`, `z = 3`.

2. Implement an inference function for the logistic regression with the signature `func inference(inputs [][]float64, w []float64, b float) []float64`.
   The function accepts a slice of $m$ input vectors $\vec{x}$ ($|\vec{x}| = n$), a vector with weights $\vec{w}$, ($|\vec{w}| = n$) and a bias $b$,
   and outputs a vector of inference probabilities for the logistic regression algorithm $\vec{p}_{\+}$.

   $|\vec{p}_{\+}| = m$.

   As a helper function, implement `func dot(a []float64, b []float64) float64` function that calculates a dot product of two input vectors.

   Add unit tests for the functions `dot` and `inference`.

## Part II

3. Implement a function that calculates the cost function gradient consisting of values $\frac{\partial J(\vec{w}, b)}{\partial w_j}$ and $\frac{\partial J(\vec{w}, b)}{\partial b}$:

   ```go
   func dCost(inputs [][]float64, y, p []float64) (dw []float64, db float64)
   ```

   The inputs are

   1. a slice of $m$ input vectors $\vec{x}$ ($|\vec{x}| = n$)
   2. a vector of inference probabilities $\vec{p}_{\+}$
   
      ($|\vec{p}_{\+}| = m$)
   5. a vector of target values $\vec{y}$ ($|\vec{y}| = m$).

   Implement unit tests for the function.

4. Implement a gradient descent algorithm for the data set in the file `data/exams1.csv`.

   The algorithm utilizes the gradient calculation

   $$ w_j = w_j - \alpha \frac{\partial J(\vec{w}, b)}{\partial w_j},\ 1 \leq j \leq n$$
   $$ b = b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b}$$

   You may rework the gradient descent algorithm implemented in [gradient-descent](https://github.com/prog-1/gradient-descent) and [gradient-descent-2](https://github.com/prog-1/gradient-descent-2) homeworks.
