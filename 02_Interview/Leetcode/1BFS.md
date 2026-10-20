<!-- zhiye:{"title":"广度优先搜索 BFS","tags":["算法","面试"],"visibility":"public","folderPath":["公开笔记","算法面试","搜索与回溯"],"pinned":false} -->
常用模版
```python
queue1=collections.deque()
if root:
	queue1.append(root)

while queue1:
	tmp=[]
	for _ in range(len(queue)):
		node=queue.popleft()
		tmp.append(node.val)
		if node.left:
			queue.append(node.left)
	res.append(tmp)
	tmp.appendleft(node.val) # 偶数层 -> 插入队列头
	
return res[::-1]
```

Python 中使用 collections 中的双端队列 deque () ，其 popleft () 方法可达到 O (1) 时间复杂度；列表 list 的 pop (0) 方法时间复杂度为 O (N) 。



```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        edges = collections.defaultdict(list)
        indeg = [0] * numCourses

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1
        
        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
        visited = 0
        while q:
            visited += 1
            u = q.popleft()
            for v in edges[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return visited == numCourses
```

