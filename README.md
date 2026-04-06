#  MERGRAD

**Mergrad** is a lightweight library for automatic differentiation.


### Usage

The core building block is an instance of the `Value` class.
Each `Value` object lets you build a computation graph and then obtain gradients from it.


```
x = Value(2.)
y = x ** 2
```

When you use `Value` objects in arithmetic expressions, the computation graph is built automatically.

![some text](https://github.com/gitmskhl/mergrad/blob/main/images/im1.png)


To compute gradients, call the `.backward()` method on the variable that stores the result of the computation.

```
x = Value(2.)
y = x ** 2
y.backward()
```
The computation graph looks like this:

![some text](https://github.com/gitmskhl/mergrad/blob/main/images/im2.png)


After calling `.backward()`, gradient values are stored in the `.grad` field.

```
from mergrad.functional import sin

x = Value(1.6)
y = Value(2.5)
z = sin(x ** 2 + x * y)
z.backward()
print(x.grad)
print(y.grad)
```

Output:

```
5.483005781936319
1.5390893422979142
```

![some text](https://github.com/gitmskhl/mergrad/blob/main/images/im3.png)
