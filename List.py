class Node:
    
    def __init__(self, left = None, data = None, right = None):
        self.left = left
        self.data = data
        self.right = right
        
    def __repr__(self):
        return self.data
    
    def __add__(self,other):
        return self.data+other.data
    
    def __mul__(self,other):
        return self.data*other.data
        
class List:
    
    def __init__(self):
        self.right = None
        self.left = None
        self.count = 0
        
    def append(self, data):
        if self.right is None and self.left is None:
            new = Node(data = data)
            self.right = self.left = new
        else:
            new = Node(self.right, data = data)
            self.right.right = new
            self.right = self.right.right
            
    def lappend(self,data):
        if self.right is None and self.left is None:
            new = Node(data = data)
            self.right = self.left = new
        else:
            new = Node(None, data, self.left)
            self.left.left = new
            self.left = self.left.left
