def f(x):
    return (a * x) // b - a * (x // b)
a, b, n = map(int, input().split())
k = (n + 1) // b
x = max(k * b - 1, 0)
ans = f(x)
ans = max(ans, f(n))
print(ans)