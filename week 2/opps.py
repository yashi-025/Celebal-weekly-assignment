class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        
class Single_LinkedList:
    def __init__(self):
        self.head = None
        
    def add_node(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return  # return help in creating a clean list

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
            
    def print_list(self):
        if self.head is None:
              print("Single Linked list is empty!!")
              return
                    
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("Link end")
        
    def delete_nth_node(self, n):
        try:
            if self.head is None:
                raise IndexError("Can't delete from empyt list.")
            if n <= 0:
                 raise IndexError("Index should be positive integer and greater than 1.")  
            if n == 1:
                print(f"Delete node with data: {self.head.data}, position: {n}")
                self.head = self.head.next
                return
            
            current = self.head
            for i in range(n-2):
                if current is None or current.next is None:
                    raise IndexError("Index out of range")
                current = current.next
                
            if current is None:
                raise IndexError("Index out of range")                    
                    
            print(f"Delete node with data: {current.next.data}, position: {n}")
            current.next = current.next.next
        except IndexError as e:
            print(f"Error: {e}")
        
if __name__ == "__main__":
    Linklist = Single_LinkedList()  
     
    Linklist.print_list()  # printing empty list
    
    Linklist.add_node(10)
    Linklist.add_node(20)  
    Linklist.add_node(30)
    Linklist.add_node(40)
    Linklist.add_node(50)
    Linklist.add_node(60)      
    
    print("List after input:")
    Linklist.print_list()
    
    print("delete 4th node")
    Linklist.delete_nth_node(4)
    Linklist.print_list()
    
    # to try fault cases and other cases. 
    Linklist.delete_nth_node(14)
