注意输入和输出
```python
a,b=input().split()
lisa=list(a.split("."))
lisb=list(b.split("."))
n = int(input())
dp = []
for i in range(n):
    dp.append(list(map(int, input().split(','))))  # 每行输入用逗号分隔
print("{:.10f}".format(2 / (a * (a - 1))))
print(f"{2 / (a * (a - 1)):.10f}")
print(round(2 / (a * (a - 1)), 10))

```

```python

import sys
import collections

n, q = map(int, input().split())
key = set(input().split())
res = collections.defaultdict(int)

names = []  # 保存输入顺序
for i in range(n):
    name, m = input().split()
    sh = set(input().split())
    res[name] = len(sh & key)
    names.append(name)

# 按关键词数量排序，数量相同时保持输入顺序
a = sorted(names, key=lambda x: res[x],reverse=True)
# # 按 value 排序，如果相同按 name 排序
# # a = sorted(res.items(), key=lambda x: (x[1], x[0]),reverse=True)
for name in a:
    print(name)

```

```python

q=int(input())
ans=[]
for _ in range(q):
    a=list(input().split())
    if a[0]=='1':
        ans.append(int(a[1]))
    elif a[0]=='2':
        if ans:
            ans.pop()
    elif a[0]=='3':
        print(ans[int(a[1])])
    elif a[0]=='4':
        ans.insert(int(a[1])+1,int(a[2]))
    elif a[0]=='5':
        ans.sort()
    elif a[0]=='6':
        ans=sorted(ans,reverse=True)
    elif a[0]=='7':
        print(len(ans))
    elif a[0]=='8':
        print(*ans)

```

```python
import bisect

class OrderedSet:
    def __init__(self):
        self.data = []  # 使用有序列表存储元素

    def insert(self, x):
        i = bisect.bisect_left(self.data, x)
        if i == len(self.data) or self.data[i] != x:  # 不存在才插入
            self.data.insert(i, x)

    def erase(self, x):
        i = bisect.bisect_left(self.data, x)
        if i < len(self.data) and self.data[i] == x:
            self.data.pop(i)

    def contains(self, x):
        i = bisect.bisect_left(self.data, x)
        return i < len(self.data) and self.data[i] == x

    def size(self):
        return len(self.data)

    def predecessor(self, x):
        i = bisect.bisect_left(self.data, x)
        if i > 0:
            return self.data[i-1]
        return -1

    def successor(self, x):
        i = bisect.bisect_right(self.data, x)
        if i < len(self.data):
            return self.data[i]
        return -1


def main():
    n = int(input())
    s = OrderedSet()
    for _ in range(n):
        parts = list(map(int, input().split()))
        op = parts[0]
        if op == 1:
            s.insert(parts[1])
        elif op == 2:
            s.erase(parts[1])
        elif op == 3:
            print("YES" if s.contains(parts[1]) else "NO")
        elif op == 4:
            print(s.size())
        elif op == 5:
            print(s.predecessor(parts[1]))
        elif op == 6:
            print(s.successor(parts[1]))


if __name__ == "__main__":
    main()

```