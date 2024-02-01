Logistic regression

1. Implement a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function with a signature `func sigmoid(z float64) float64`.
   Add unit tests for the function. Verify the values for `z = -1`, `z = 0`, `z = 1`, `z = 2`, `z = 3`.

2. Implement an inference function for the logistic regression with the signature `func inference(inputs [][]float64, w []float64, b float) []float64`.
   The function accepts a slice of $m$ input vectors $\vec{x}$ ($|\vec{x}| = n$), a vector with weights $\vec{w}$, ($|\vec{w}| = n$) and a bias $b$,
   and outputs a vector of inference probabilities for the logistic regression algorithm $\vec{p}_{\+}$.

   $|\vec{p}_{\+}| = m$.

   As a helper function, implement `func dot(a []float64, b []float64) float64` function that calculates a dot product of two input vectors.
