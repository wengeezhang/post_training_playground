def number_generator(n):
    """生成从0到n-1的数字"""
    for i in range(n):
        print(f"yielding {i}")
        yield i
        print(f"yielded {i}")

# 使用生成器
gen = number_generator(5)

print("first 5 numbers:")

print("start 1")
print(next(gen))
print("start 2")
print(next(gen))
print("start 3")
print(next(gen))
print("start 4")
print(next(gen))
print("start 5")
print(next(gen))
print("done")
